import torch
import numpy as np
import os
import cv2
import OpenEXR
import Imath
from PIL import Image

def collate_fn(list_data):
    cam_pose, depth_im, _ = list_data
    # Concatenate all lists
    return cam_pose, depth_im, _


class ScanNetDataset(torch.utils.data.Dataset):
    """Pytorch Dataset for a single scene. getitem loads individual frames"""
    ## attention
    def __init__(self, n_imgs, scene, data_path, max_depth, min_depth, id_list=None, datatype= "mountain"):
        """
        Args:
        """
        self.n_imgs = n_imgs
        self.scene = scene
        self.data_path = data_path
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.datatype = datatype
        if id_list is None:
            self.id_list = [i for i in range(1,n_imgs)]
        else:
            self.id_list = id_list

    def __len__(self):
        return self.n_imgs-1

    def __getitem__(self, id):
        """
        Returns:
            dict of meta data and images for a single frame
        """
        id = self.id_list[id]
        # id = id+1
        if self.datatype == "mountain":
            cam_pose = np.loadtxt(os.path.join(self.data_path, self.scene, "pose", "extrinsic_"+str(id) + ".txt"), delimiter=',')

            # Read depth image and camera pose
            # depth_im = cv2.imread(os.path.join(self.data_path, self.scene, "depth", str(id)+"_depth0022" + ".exr"), -1).astype(
            #     np.float32)

            ## Read depth image with exr formate
            File = OpenEXR.InputFile(os.path.join(self.data_path, self.scene, "depth", str(id)+"_depth0001" + ".exr"))
            PixType = Imath.PixelType(Imath.PixelType.FLOAT)
            DW = File.header()['dataWindow']
            Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
            rgb = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
            r = np.reshape(rgb[0], (Size[1], Size[0]))
            depth_im = np.zeros((Size[1], Size[0]), dtype=np.float32)
            depth_im = r
            depth = np.array(depth_im)

            ##depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
            depth[depth > self.max_depth] = 0
            depth[depth < self.min_depth] = 0

            # Read RGB image
            # color_image = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, self.scene, "color", str(id) + ".jpg")),
            #                            cv2.COLOR_BGR2RGB)

            color_image = cv2.imread(os.path.join(self.data_path, self.scene, "color", str(id) + ".jpg"))
            #color_image = cv2.resize(color_image, (depth_im.shape[1], depth_im.shape[0]), interpolation=cv2.INTER_AREA)


        if self.datatype == "indoor":
            if id <10:
                name = "000"+str(id)+"00"
            else:
                name = "00"+str(id)+"00"
            cam_pose = np.loadtxt(os.path.join(self.data_path, self.scene, "pose", name + ".txt"))

            # Read depth image and camera pose
            depth = cv2.imread(os.path.join(self.data_path, self.scene, "depth", name + ".png"), -1).astype(
                np.float32)
            print(np.sum(depth))
            depth /= 1000.  # depth is saved in 16-bit PNG in millimeters
            print("depth",np.max(depth),np.min(depth))
            depth[depth > self.max_depth] = 0

            # Read RGB image
            color_image = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, self.scene, "color", name+ ".jpg")),
                                       cv2.COLOR_BGR2RGB)
            color_image = cv2.resize(color_image, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)

        return cam_pose, depth, color_image
