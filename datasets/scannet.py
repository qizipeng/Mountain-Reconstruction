import os
import numpy as np
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset
import OpenEXR
import Imath
from skimage import measure
import trimesh

datatype = "mountain" ##mountain
class ScanNetDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)

        assert self.mode in ["train", "valid", "test"]
        self.metas = self.build_list()
        if mode == 'test':
            self.source_path = 'scans_test'
        elif mode =='valid':
            self.source_path = 'scans_val'
        else:
            self.source_path = 'scans'

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100

    def build_list(self):
        metas = []
        dirlist = os.listdir(os.path.join(self.datapath,self.tsdf_file,'1',self.mode))
        print(os.path.join(self.datapath,self.tsdf_file,'1',self.mode))
        print(dirlist)
        for sence_name in dirlist:
            # print(os.path.join(self.datapath, self.tsdf_file, '1', self.mode, sence_name, 'fragments.pkl'))
            with open(os.path.join(self.datapath, self.tsdf_file, '1', self.mode, sence_name, 'fragments.pkl'), 'rb') as f:
                metas += pickle.load(f)
        return metas


        # with open(os.path.join(self.datapath, self.tsdf_file, "1", self.mode, 'object6', 'fragments.pkl'), 'rb') as f:
        #     metas = pickle.load(f)
        # return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid):
        if datatype == "mountain":
            intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic.txt'), delimiter=',')[:3, :3]
            intrinsics = intrinsics.astype(np.float32)
            # print(os.path.join(filepath, 'pose', 'extrinsic_{}.txt'.format(str(vid))))
            extrinsics = np.loadtxt(os.path.join(filepath, 'pose', 'extrinsic_{}.txt'.format(str(vid))), delimiter=',')
        elif datatype =="indoor":

            intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsics_depth.txt'))[:3, :3]
            intrinsics = intrinsics.astype(np.float32)
            # print(os.path.join(filepath, 'pose', 'extrinsic_{}.txt'.format(str(vid))))
            id = vid
            if id <10:
                name = "000"+str(id)+"00"
            else:
                name = "00"+str(id)+"00"
            extrinsics = np.loadtxt(os.path.join(filepath, 'pose', name+'.txt'))
        return intrinsics, extrinsics

    def read_img(self, filepath, IsNormal = False):
        # print(filepath)
        img = Image.open(filepath)
        if IsNormal:
            img = img.resize((128,128))
        return img

    def read_depth(self, filepath,min,max):
        if datatype=="mountain":
            File = OpenEXR.InputFile(filepath)
            PixType = Imath.PixelType(Imath.PixelType.FLOAT)
            DW = File.header()['dataWindow']
            Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
            rgb = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
            r = np.reshape(rgb[0], (Size[1], Size[0]))
            depth_im = np.zeros((Size[1], Size[0]), dtype=np.float32)
            depth_im = r
            depth = np.array(depth_im)
            depth[depth > max] = 0
            depth[depth < min] = 0
        elif datatype=="indoor":
            depth = cv2.imread(filepath).astype(
                np.float32)
            depth /= 1000.  # depth is saved in 16-bit PNG in millimeters
            depth[depth > max] = 0
            depth = depth[:,:,0]
        # print(depth)
        return depth

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, "1", self.mode, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                # print(np.sum(full_tsdf.f.arr_0))
                # verts, faces, norms, vals = measure.marching_cubes(full_tsdf.f.arr_0)
                # verts = verts * (0.02* 2 ** l) + self.metas[0]['vol_origin'] # voxel grid coordinates to world coordinates
                # mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
                # mesh.export(os.path.join("./result_vis/", "mode", '{}_{}_{}.ply'.format("scene_name", l, "target")))
                # print(np.max(full_tsdf.f.arr_0), np.min(full_tsdf.f.arr_0))
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]
        imgs = []
        depth = []
        normals = []
        extrinsics_list = []
        intrinsics_list = []
        name_id = []
        tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])
        for i, vid in enumerate(meta['image_ids']):
            # name_id.append(os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}_depth0001.exr'.format(vid)))
            # load images
            id = vid
            if datatype == "mountain":
                imgs.append(
                    self.read_img(
                        os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid))))
            elif datatype == "indoor":
                if id < 10:
                    name = "000" + str(id) + "00"
                else:
                    name = "00" + str(id) + "00"
                imgs.append(
                    self.read_img(
                        os.path.join(self.datapath, self.source_path, meta['scene'], 'color', name+'.jpg')))
            # normals.append(
            #     self.read_img(
            #         os.path.join(self.datapath, self.source_path, meta['scene'], 'segments', '{}.png'.format(vid)), IsNormal = False))
            if datatype == "mountain":
                depth.append(
                    self.read_depth(
                        os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}_depth0001.exr'.format(vid)),10.,120.0)
                )
            elif datatype == "indoor":
                if id < 10:
                    name = "000" + str(id) + "00"
                else:
                    name = "00" + str(id) + "00"
                depth.append(
                    self.read_depth(
                        os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', name+'.png'),0.0,3.0)
                )

            # load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']),
                                                        vid)

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)
        # print('scannet'
        # )
        # print(depth[0])

        items = {
            'imgs': imgs,
            'depth': depth,
            # 'normals': normals,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'vol_origin': meta['vol_origin'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
            "id": idx,
        }

        if self.transforms is not None:
            items = self.transforms(items)
        return items
