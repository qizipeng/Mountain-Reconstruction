import torch
import torch.nn as nn

from .backbone import MnasMulti
from .neucon_network import NeuConNet
from .gru_fusion import GRUFusion
from utils import tocuda


class NeuralRecon(nn.Module):
    '''
    NeuralRecon main class.
    '''

    def __init__(self, cfg):
        super(NeuralRecon, self).__init__()
        self.cfg = cfg.MODEL
        self.IsNeuralRecon = cfg.NeuralRecon
        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.n_scales = len(self.cfg.THRESHOLDS) - 1

        # networks
        self.backbone2d = MnasMulti(alpha, IsNeuralRecon = cfg.NeuralRecon)
        self.neucon_net = NeuConNet(cfg.MODEL, cfg.NeuralRecon)
        # for fusing to global volume
        self.fuse_to_global = GRUFusion(cfg.MODEL, direct_substitute=True)

    def reset_volume(self):
        self.fuse_to_global.reset_volume()
        #self.neucon_net.reset_volume()
    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        # return x/255.
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def forward(self, inputs, save_mesh=False):
        '''

        :param inputs: dict: {
            'imgs':                    (Tensor), images,
                                    (batch size, number of views, C, H, W)
            'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                    (batch size, number of views, 4, 4)
            'proj_matrices':           (Tensor), projection matrix,
                                    (batch size, number of views, number of scales, 4, 4)
            when we have ground truth:
            'tsdf_list':               (List), tsdf ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            'occ_list':                (List), occupancy ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            others: unused in network
        }
        :param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
            When it comes to save results:
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
        }
                 loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
            'total_loss':              (Tensor), total loss
        }
        '''
        inputs = tocuda(inputs)
        outputs = {}
        global_mesh = {}
        imgs = torch.unbind(inputs['imgs'], 1)#9 b 3 512 512
        # normals = torch.unbind(inputs['normals'],1)
        # image feature extraction
        # in: images; out: feature maps
        # juleiloss = None
        if not self.IsNeuralRecon:
            # features = [self.backbone2d(self.normalizer(img),self.IsNeuralRecon) for img in imgs]  ##qizipeng:resnet
            # juleiloss = self.backbone2d.juleiloss / len(imgs)
            import torch.nn.functional as F
            # features = [F.cross_entropy(self.backbone2d(self.normalizer(img), self.IsNeuralRecon), torch.zeros((1, 128, 128)).cuda().long()) for img in imgs]
            features = [self.backbone2d(self.normalizer(img), self.IsNeuralRecon) for img in imgs]
            # juleiloss1 = [x[1] for x in features]
            # juleiloss1 = torch.stack(juleiloss1)
            # juleiloss1 = torch.sum(juleiloss1)/len(imgs)
            #
            # # juleiloss2 = [x[1] for x in features]
            # # juleiloss2 = torch.stack(juleiloss2)
            # # juleiloss2 = torch.sum(juleiloss2)/len(imgs)
            # #
            # # juleiloss3 = [x[1] for x in features]
            # # juleiloss3 = torch.stack(juleiloss3)
            # # juleiloss3 = torch.sum(juleiloss3)/len(imgs)
            #
            # juleiloss = juleiloss1#+juleiloss2+juleiloss3)#/3.
            # features = [x[0] for x in features]

        else:
            features = [self.backbone2d(self.normalizer(img)) for img in imgs]  ##qizipeng:resnet
        # for f in features:
        #     for x in f:
        #         panduan = torch.isnan(x)
        #         print(torch.sum(panduan))


        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs, loss_dict = self.neucon_net(features, inputs, outputs)
        # fuse to global volume.
        if not self.training and 'coords' in outputs.keys():
            print("yes")
            outputs = self.fuse_to_global(outputs['coords'], outputs['tsdf'], inputs, self.n_scales, outputs, save_mesh,True )

        # if 'coords' in outputs.keys():
        #     outputs = self.fuse_to_global(outputs['coords'], outputs['tsdf'], inputs, self.n_scales, outputs, save_mesh)

        # gather loss.
        print_loss = 'Loss: '
        for k, v in loss_dict.items():
            print_loss += f'{k}: {v} '

        weighted_loss = 0

        for i, (k, v) in enumerate(loss_dict.items()):
            weighted_loss += v * self.cfg.LW[i]
        # loss_dict.update({'julei_loss': juleiloss})
        # if juleiloss is not None:
        #     weighted_loss = weighted_loss + juleiloss
        loss_dict.update({'total_loss': weighted_loss})
        return outputs, loss_dict#, global_mesh
