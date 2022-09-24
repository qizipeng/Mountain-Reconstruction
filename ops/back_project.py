import torch
from torch.nn.functional import grid_sample
import torch.nn as nn

class Back_project(nn.Module):
    def __init__(self,channel,n_view = 9):
        super(Back_project, self).__init__()
        self.point_transforms = nn.Sequential(
            nn.Linear(channel, n_view),
            nn.BatchNorm1d(n_view),
            nn.ReLU(True),
        )



    def back_project(self , coords, origin, voxel_size, feats, KRcam, IsNeuralRecon = False):
        '''
        Unproject the image fetures to form a 3D (sparse) feature volume

        :param coords: coordinates of voxels,
        dim: (num of voxels, 4) (4 : batch ind, x, y, z)
        :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
        dim: (batch size, 3) (3: x, y, z)
        :param voxel_size: floats specifying the size of a voxel
        :param feats: image features
        dim: (num of views, batch size, C, H, W)
        :param KRcam: projection matrix
        dim: (num of views, batch size, 4, 4)
        :return: feature_volume_all: 3D feature volumes
        dim: (num of voxels, c + 1)
        :return: count: number of times each voxel can be seen
        dim: (num of voxels,)
        '''
        n_views, bs, c, h, w = feats.shape

        feature_volume_all = torch.zeros(coords.shape[0], c + 1).cuda()
        count = torch.zeros(coords.shape[0]).cuda()

        for batch in range(bs):
            batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
            coords_batch = coords[batch_ind][:, 1:]

            coords_batch = coords_batch.view(-1, 3)##n-points 3
            origin_batch = origin[batch].unsqueeze(0)
            feats_batch = feats[:, batch]  ##n-views c h w
            proj_batch = KRcam[:, batch]

            grid_batch = coords_batch * voxel_size + origin_batch.float()
            rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)##n-views n-points 3
            rs_grid = rs_grid.permute(0, 2, 1).contiguous()##n-views 3 n-points
            nV = rs_grid.shape[-1]
            rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1) ##n-views 4 n-points  4:(x,y,z,1)

            # Project grid
            im_p = proj_batch @ rs_grid
            im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
            im_x = im_x / im_z
            im_y = im_y / im_z   ####这里 /im_z 搞清楚为什么

            im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)##n-view 1 n-points 2
            ####计算每个视角权重的权重： qizipeng add
            ###计算像平面上每个点对应的角度，像平面与相机中心组成一个四棱锥。权重为相机中心与像平面的点组成角度的cos值（或者用x^2+y^2 试试 因为 焦距是一致的）
            if not IsNeuralRecon:
                #print("backproject")
                img_x_2 = im_x * im_x ##n-view 1 n-points
                img_y_2 = im_y * im_y

                ##qzp add feature fusion
                weight = torch.sqrt(img_x_2 + img_y_2) ##n-view 1 n-points
                weight = 1/(weight + torch.tensor(0.0000001)).cuda()

            mask = im_grid.abs() <= 1
            mask = (mask.sum(dim=-1) == 2) & (im_z > 0) ##n-view 1 n-points

            feats_batch = feats_batch.view(n_views, c, h, w)
            im_grid = im_grid.view(n_views, 1, -1, 2)   ###n-views 1 n-points 2
            features = grid_sample(feats_batch, im_grid, mode = 'bilinear',padding_mode='zeros', align_corners=True)

            ##qzp add remove nan
            # features = torch.nan_to_num(features)
            features[torch.isnan(features)==True]=0

            features = features.view(n_views, c, -1)  ##n-views c n-points
            mask = mask.view(n_views, -1)
            ###qizipeng add  feature fusion
            if not IsNeuralRecon:
                weight = weight.view(n_views,-1)
                weight = mask * weight
                ##qzp add remove nan
                weight[torch.isnan(weight) == True] = 0

                weight = torch.softmax(weight,0)
                weight = weight.unsqueeze(1)   ##n-views 1  n-points

            im_z = im_z.view(n_views, -1)
            # remove nan
            features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0

            ###qizipeng add feature fusion
            features_pre = features           ##n-view c n-point
            if not IsNeuralRecon:
                features = features * weight   ##n-view c n-point
            im_z[mask == False] = 0

            count[batch_ind] = mask.sum(dim=0).float()  #1 n-points

            # aggregate multi view

            features = features.sum(dim=0) # c n-points
            if not IsNeuralRecon:
                features = features.permute(1, 0).contiguous() ##n-point c
                learned_weight = self.point_transforms(features) ##n-points n_view
                learned_weight = learned_weight.permute(1,0).contiguous() ##n_view n-ponts
                learned_weight = learned_weight.view(n_views, -1)
                learned_weight = mask * learned_weight
                ##qzp add remove nan
                learned_weight[torch.isnan(learned_weight) == True] = 0

                learned_weight = torch.softmax(learned_weight, 0) ##n_view n-ponts
                learned_weight = learned_weight.unsqueeze(1) ##n_view 1 n-ponts

                features = features_pre * learned_weight
                features = features.sum(dim=0)  # c n-points
            mask = mask.sum(dim=0)
            invalid_mask = mask == 0
            mask[invalid_mask] = 1
            in_scope_mask = mask.unsqueeze(0)

            ##qizipeng delete original code
            if IsNeuralRecon:
                features /= in_scope_mask
            features = features.permute(1, 0).contiguous()

            # concat normalized depth value
            im_z = im_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous()
            im_z_mean = im_z[im_z > 0].mean()
            im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
            im_z_norm = (im_z - im_z_mean) / im_z_std
            im_z_norm[im_z <= 0] = 0

            features = torch.cat([features, im_z_norm], dim=1)

            feature_volume_all[batch_ind] = features
        return feature_volume_all, count
