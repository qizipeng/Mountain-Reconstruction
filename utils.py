import os
import torch
from torch.nn import init
import trimesh
import numpy as np
import torchvision.utils as vutils
from skimage import measure
from loguru import logger
from tools.render import Visualizer
import cv2
import open3d as o3d
from tools.tsdf_fusion.fusion import *

# print arguments
def print_args(args):
    logger.info("################################  args  ################################")
    for k, v in args.__dict__.items():
        logger.info("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    logger.info("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        if len(vars.shape) == 0:
            return vars.data.item()
        else:
            return [v.data.item() for v in vars]
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def save_scalars(logger, mode, scalar_dict, metrics = None, epoch_idx=0):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, epoch_idx)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], epoch_idx)
    if metrics is not None:
        metrics = tensor2float(metrics)
        for key, value in metrics.items():
            if not isinstance(value, (list, tuple)):
                name = '{}/{}'.format(mode, key)
                logger.add_scalar(name, value, epoch_idx)
            else:
                for idx in range(len(value)):
                    name = '{}/{}_{}'.format(mode, key, idx)
                    logger.add_scalar(name, value[idx], epoch_idx)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}

    def reset(self):
        self.data = {}
        self.count = 0


def coordinates(voxel_dim, device=torch.device('cuda')):
    """ 3d meshgrid of given size.

    Args:
        voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume

    Returns:
        torch long tensor of size (3,nx*ny*nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


def apply_log_transform(tsdf):
    sgn = torch.sign(tsdf)
    out = torch.log(torch.abs(tsdf) + 1)
    out = sgn * out
    return out


def sparse_to_dense_torch_batch(locs, values, dim, default_val):
    dense = torch.full([dim[0], dim[1], dim[2], dim[3]], float(default_val), device=locs.device)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], locs[:, 3]] = values
    return dense


def sparse_to_dense_torch(locs, values, dim, default_val, device):
    # print(values)
    dense = torch.full([dim[0], dim[1], dim[2]], float(default_val), device=device)
    # print(dense.shape)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_channel(locs, values, dim, c, default_val, device):
    dense = torch.full([dim[0], dim[1], dim[2], c], float(default_val), device=device)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_np(locs, values, dim, default_val):
    dense = np.zeros([dim[0], dim[1], dim[2]], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


class SaveScene(object):
    def __init__(self, cfg):
        self.cfg = cfg
        log_dir = cfg.LOGDIR.split('/')[-1]
        self.log_dir = os.path.join('results', 'scene_' + cfg.DATASET + '_' + log_dir)
        self.scene_name = None
        self.global_origin = None
        self.tsdf_volume = []  # not used during inference.
        self.weight_volume = []

        self.coords = None

        self.keyframe_id = None

        if cfg.VIS_INCREMENTAL:
            self.vis = Visualizer()

    def close(self):
        self.vis.close()
        cv2.destroyAllWindows()

    def reset(self):
        self.keyframe_id = 0
        self.tsdf_volume = []
        self.weight_volume = []

        # self.coords = coordinates(np.array([416, 416, 128])).float()

        # for scale in range(self.cfg.MODEL.N_LAYER):
        #     s = 2 ** (self.cfg.MODEL.N_LAYER - scale - 1)
        #     dim = tuple(np.array([416, 416, 128]) // s)
        #     self.tsdf_volume.append(torch.ones(dim).cuda())
        #     self.weight_volume.append(torch.zeros(dim).cuda())

    @staticmethod
    def tsdf2mesh(voxel_size, origin, tsdf_vol):
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)
        verts = verts * voxel_size + origin  # voxel grid coordinates to world coordinates
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
        return mesh

    def vis_incremental(self, epoch_idx, batch_idx, imgs, outputs):
        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()
        if self.cfg.DATASET == 'demo':
            origin[2] -= 1.5

        if (tsdf_volume == 1).all():
            logger.warning('No valid partial data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            # vis
            key_frames = []
            for img in imgs[::3]:
                img = img.permute(1, 2, 0)
                img = img[:, :, [2, 1, 0]]
                img = img.data.cpu().numpy()
                img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                key_frames.append(img)
            key_frames = np.concatenate(key_frames, axis=0)
            cv2.imshow('Selected Keyframes', key_frames / 255)
            cv2.waitKey(1)
            # vis mesh
            self.vis.vis_mesh(mesh)

    def save_incremental(self, epoch_idx, batch_idx, imgs, outputs):
        save_path = os.path.join('incremental_' + self.log_dir + '_' + str(epoch_idx), self.scene_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()
        if self.cfg.DATASET == 'demo':
            origin[2] -= 1.5

        if (tsdf_volume == 1).all():
            logger.warning('No valid partial data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            # save
            mesh.export(os.path.join(save_path, 'mesh_{}.ply'.format(self.keyframe_id)))

    def save_scene_eval(self, epoch, outputs, batch_idx=0):
        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()

        if (tsdf_volume == 1).all():
            logger.warning('No valid data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            # save tsdf volume for atlas evaluation
            data = {'origin': origin,
                    'voxel_size': self.cfg.MODEL.VOXEL_SIZE,
                    'tsdf': tsdf_volume}
            save_path = '{}_fusion_eval_{}'.format(self.log_dir, epoch)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.savez_compressed(
                os.path.join(save_path, '{}.npz'.format(self.scene_name)),
                **data)
            mesh.export(os.path.join(save_path, '{}.ply'.format(self.scene_name)))

    def __call__(self, outputs, inputs, epoch_idx):
        # no scene saved, skip
        if "scene_name" not in outputs.keys():
            return

        batch_size = len(outputs['scene_name'])
        for i in range(batch_size):
            scene = outputs['scene_name'][i]
            self.scene_name = scene.replace('/', '-')

            if self.cfg.SAVE_SCENE_MESH:
                self.save_scene_eval(epoch_idx, outputs, i)

def tsdf2mesh(voxel_size, origin, tsdf_vol):
    print(np.max(tsdf_vol),np.min(tsdf_vol), np.mean(tsdf_vol))
    verts, faces, norms, vals = measure.marching_cubes(tsdf_vol)
    verts = verts * voxel_size + origin  # voxel grid coordinates to world coordinates
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
    return mesh


def tedf2point(voxel_size, origin, tsdf_vol):
    verts = measure.marching_cubes(tsdf_vol, level=0)[0]
    verts_ind = np.round(verts).astype(int)
    verts = verts * voxel_size + origin

    # Get vertex colors
    rgb_vals = 255
    colors_b = np.floor(rgb_vals / 256*256)
    colors_g = np.floor((rgb_vals - colors_b * 256*256) / 256)
    colors_r = rgb_vals - colors_b * 256*256 - colors_g * 256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    pc = np.hstack([verts, colors])
    return pc

def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's

    Returns:
        ([indices], [distances])

    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def caculate_metrics(tsdf_pre, tsdf_target, orgin, origin_target, volxsize, scene_name ,threshold=.05, down_sample = .02, epoch_idx = 0,mode = 'train'):

    # print(tsdf_target[0].shape, torch.sum(tsdf_target[0]))
    tsdf_pre = tsdf_pre[0].data.cpu().numpy()
    tsdf_target = tsdf_target[0].data.cpu().numpy()
    origin = orgin[0].data.cpu().numpy()
    origin_target = origin_target[0].data.cpu().numpy()
    if (tsdf_pre == 1).all():
        logger.warning('No valid data for scene {}'.format(scene_name))
    else:
        # Marching cubes
        # tsdf_target = tsdf_target*1000

        # point_target = tedf2point(volxsize, origin_target, tsdf_target)
        # pcwrite(os.path.join(os.path.join("./result_vis/", mode,'{}_{}_{}.ply'.format(scene_name, epoch_idx, "target"))))

        mesh_target = tsdf2mesh(volxsize, origin_target, tsdf_target)
        mesh_pre = tsdf2mesh(volxsize, origin, tsdf_pre)

        if (epoch_idx+1) % 1 ==0:
            mesh_pre.export(os.path.join("./result_vis/", mode,'{}_{}_{}.ply'.format(scene_name, epoch_idx, "pre")))
            mesh_target.export(os.path.join("./result_vis/", mode,'{}_{}_{}.ply'.format(scene_name, epoch_idx, "target")))
        pcd_pred = o3d.geometry.PointCloud()
        pcd_trgt = o3d.geometry.PointCloud()

        mesh_pre_V = np.asarray(mesh_pre.vertices)
        mesh_target_V = np.asarray(mesh_target.vertices)

        pcd_pred.points = o3d.utility.Vector3dVector(mesh_pre_V)
        pcd_trgt.points = o3d.utility.Vector3dVector(mesh_target_V)
        if down_sample:
            pcd_pred = pcd_pred.voxel_down_sample(down_sample)
            pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
        verts_pred = np.asarray(pcd_pred.points)
        verts_trgt = np.asarray(pcd_trgt.points)

        _, dist1 = nn_correspondance(verts_pred, verts_trgt)
        _, dist2 = nn_correspondance(verts_trgt, verts_pred)
        dist1 = np.array(dist1)
        dist2 = np.array(dist2)

        precision = np.mean((dist2 < threshold).astype('float'))
        recal = np.mean((dist1 < threshold).astype('float'))
        fscore = 2 * precision * recal / (precision + recal)
        metrics = {'dist1': np.mean(dist2),
                   'dist2': np.mean(dist1),
                   'prec': precision,
                   'recal': recal,
                   'fscore': fscore,
                   }
        return metrics



        # save tsdf volume for atlas evaluation

        # mesh.export(os.path.join(save_path, '{}.ply'.format(self.scene_name)))


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    return net