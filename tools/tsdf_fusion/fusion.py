# Copyright (c) 2018 Andy Zeng

import numpy as np

from numba import njit, prange
from skimage import measure
import torch


class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images.
    """

    def __init__(self, vol_bnds, voxel_size, use_gpu=True, margin=5):
        """Constructor.

        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """

        # except Exception as err:
        #     print('Warning: {}'.format(err))
        #     print('Failed to import PyCUDA. Running fusion in CPU mode.')
        #     FUSION_GPU_MODE = 0

        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = margin * self._voxel_size  # truncation on SDF
        self._color_const = 256 * 256

        # Adjust volume bounds and ensure C-order contiguous
        print('range-----------',self._vol_bnds[:, 1] - self._vol_bnds[:, 0])
        self._vol_dim = np.round((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(
            order='C').astype(int)
        print("vol_dim---------------",self._vol_dim)
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order='C').astype(np.float32)

        # Initialize pointers to voxel volume in CPU memory
        self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        ##orgin:
        # self.gpu_mode = use_gpu and FUSION_GPU_MODE
        self.gpu_mode = False
        # Copy voxel volumes to GPU
        if self.gpu_mode:
            pass
        else:
            # Get voxel grid coordinates
            xv, yv, zv = np.meshgrid(
                range(self._vol_dim[0]),
                range(self._vol_dim[1]),
                range(self._vol_dim[2]),
                indexing='ij'
            )
            self.vox_coords = np.concatenate([
                xv.reshape(1, -1),
                yv.reshape(1, -1),
                zv.reshape(1, -1)
            ], axis=0).astype(int).T

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        """Convert voxel grid coordinates to world coordinates.
        """
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        """Convert camera coordinates to pixel coordinates.
        """
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """Integrate the TSDF volume.
        """
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
        """Integrate an RGB-D frame into the TSDF volume.

        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          obs_weight (float): The weight to assign for the current observation. A higher
            value
        """
        im_h, im_w = depth_im.shape

        if color_im is not None:
            # Fold RGB color image into a single channel image
            color_im = color_im.astype(np.float32)
            color_im = np.floor(color_im[..., 2] * self._color_const + color_im[..., 1] * 256 + color_im[..., 0])
            # color_im = color_im.reshape(-1).astype(np.float32)
        else:
            color_im = np.array(0)

        if self.gpu_mode:  # GPU mode: integrate voxel volume (calls CUDA kernel)
           pass
        else:  # CPU mode: integrate voxel volume (vectorized implementation)
            # Convert voxel grid coordinates to pixel coordinates
            cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
            cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
            pix_z = cam_pts[:, 2]
            pix = self.cam2pix(cam_pts, cam_intr)
            pix_x, pix_y = pix[:, 0], pix[:, 1]

            # Eliminate pixels outside view frustum
            valid_pix = np.logical_and(pix_x >= 0,
                                       np.logical_and(pix_x < im_w,
                                                      np.logical_and(pix_y >= 0,
                                                                     np.logical_and(pix_y < im_h,
                                                                                    pix_z > 0))))
            depth_val = np.zeros(pix_x.shape)
            depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

            # print("depth_val", depth_val)
            # print("pix_z", pix_z)

            # Integrate TSDF
            depth_diff = depth_val - pix_z

            # print("depth_diff", depth_diff)

            valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
            #print('valid_pts:',valid_pts.sum())
            dist = np.minimum(1, depth_diff / self._trunc_margin)
            # print(np.sum(dist))
            # print(dist)

            valid_vox_x = self.vox_coords[valid_pts, 0]
            valid_vox_y = self.vox_coords[valid_pts, 1]
            valid_vox_z = self.vox_coords[valid_pts, 2]
            w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            valid_dist = dist[valid_pts]
            tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
            self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
            self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new
            #print('tsdf_cpu:',(self._tsdf_vol_cpu!=1).sum())

            # Integrate color
            #qizipeng delete
            old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            old_b = np.floor(old_color / self._color_const)
            old_g = np.floor((old_color - old_b * self._color_const) / 256)
            old_r = old_color - old_b * self._color_const - old_g * 256
            new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_b = np.floor(new_color / self._color_const)
            new_g = np.floor((new_color - new_b * self._color_const) / 256)
            new_r = new_color - new_b * self._color_const - new_g * 256
            new_b = np.minimum(255., np.round((w_old * old_b + obs_weight * new_b) / w_new))
            new_g = np.minimum(255., np.round((w_old * old_g + obs_weight * new_g) / w_new))
            new_r = np.minimum(255., np.round((w_old * old_r + obs_weight * new_r) / w_new))
            self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = 255# new_b * self._color_const + new_g * 256 + new_r

    def get_volume(self):
        if self.gpu_mode:
            self.cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
            self.cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
            self.cuda.memcpy_dtoh(self._weight_vol_cpu, self._weight_vol_gpu)
        return self._tsdf_vol_cpu, self._color_vol_cpu, self._weight_vol_cpu

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume.
        """
        tsdf_vol, color_vol, weight_vol = self.get_volume()

        # Marching cubes
        verts = measure.marching_cubes(tsdf_vol, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes.
        """
        tsdf_vol, color_vol, weight_vol = self.get_volume()
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol)
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin  # voxel grid coordinates to world coordinates
        # print("voxel_size:",self._voxel_size)

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud.
    """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image
    """
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    print(max_depth)
    view_frust_pts = np.array([
        (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /cam_intr[0, 0],
        (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /cam_intr[1, 1],
        np.array([0, max_depth, max_depth, max_depth, max_depth])
    ])
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file.
    """
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
            verts[i, 0], verts[i, 1], verts[i, 2],
            norms[i, 0], norms[i, 1], norms[i, 2],
            colors[i, 0], colors[i, 1], colors[i, 2],
        ))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file.
    """
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))


def integrate(
        depth_im,
        cam_intr,
        cam_pose,
        obs_weight,
        world_c,
        vox_coords,
        weight_vol,
        tsdf_vol,
        sdf_trunc,
        im_h,
        im_w,
):
    # Convert world coordinates to camera coordinates
    world2cam = torch.inverse(cam_pose)
    cam_c = torch.matmul(world2cam, world_c.transpose(1, 0)).transpose(1, 0).float()

    # Convert camera coordinates to pixel coordinates
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    pix_z = cam_c[:, 2]
    pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long()
    pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long()

    # Eliminate pixels outside view frustum
    valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
    depth_val = torch.zeros(pix_x.shape)
    depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]



    # Integrate tsdf
    depth_diff = depth_val - pix_z

    # dist = torch.minimum(1, depth_diff / sdf_trunc)
    dist = torch.clamp(depth_diff / sdf_trunc, max=1)
    # print(torch.sum(dist))
    valid_pts = (depth_val > 0) & (depth_diff >= -sdf_trunc)

    valid_vox_x = vox_coords[valid_pts, 0]
    valid_vox_y = vox_coords[valid_pts, 1]
    valid_vox_z = vox_coords[valid_pts, 2]
    # valid_vox_x = valid_vox_x[valid_pts]
    # valid_vox_y = valid_vox_y[valid_pts]
    # valid_vox_z = valid_vox_z[valid_pts]
    valid_dist = dist[valid_pts]
    w_old = weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    tsdf_vals = tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    w_new = w_old + obs_weight
    tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (w_old * tsdf_vals + obs_weight * valid_dist) / w_new
    weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new

    return weight_vol, tsdf_vol

def integrate_bak(
        depth_im,
        cam_intr,
        cam_pose,
        obs_weight,
        world_c,
        vox_coords,
        weight_vol,
        tsdf_vol,
        sdf_trunc,
        im_h,
        im_w,
):
    # Convert world coordinates to camera coordinates


    world2cam = torch.inverse(cam_pose)
    # print(world_c)

    cam_c = torch.matmul(world2cam, world_c.transpose(1, 0)).transpose(1, 0).float()

    # Convert camera coordinates to pixel coordinates
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    pix_z = cam_c[:, 2]
    pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long()
    pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long()

    # Eliminate pixels outside view frustum
    valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
    depth_val = torch.zeros(pix_x.shape)
    depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

    # Integrate tsdf
    # print("depth_val",depth_val)
    # print("pix_z", pix_z)
    depth_diff = depth_val - pix_z

    # print("depth_diff", depth_diff)

    valid_pts = (depth_val > 0) & (depth_diff >= -sdf_trunc)
    dist = torch.clamp(depth_diff / sdf_trunc, max=1)
    valid_vox_x = vox_coords[valid_pts, 0]
    valid_vox_y = vox_coords[valid_pts, 1]
    valid_vox_z = vox_coords[valid_pts, 2]

    w_old = weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    tsdf_vals = tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    valid_dist = dist[valid_pts]

    w_new = w_old + obs_weight
    tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (w_old * tsdf_vals + obs_weight * valid_dist) / w_new
    weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new

    return weight_vol, tsdf_vol







class TSDFVolumeTorch:
    """Volumetric TSDF Fusion of RGB-D Images.
    """

    def __init__(self, voxel_dim, origin, voxel_size, margin=3):
        """Constructor.

        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        # else:
        #     print("[!] No GPU detected. Defaulting to CPU.")
        self.device = torch.device("cpu")

        # Define voxel volume parameters
        self._voxel_size = float(voxel_size)
        self._sdf_trunc = margin * self._voxel_size
        self._const = 256 * 256
        self._integrate_func = integrate

        # Adjust volume bounds
        self._vol_dim = voxel_dim.long()
        self._vol_origin = origin
        self._num_voxels = torch.prod(self._vol_dim).item()

        # Get voxel grid coordinates
        xv, yv, zv = torch.meshgrid(
            torch.arange(0, self._vol_dim[0]),
            torch.arange(0, self._vol_dim[1]),
            torch.arange(0, self._vol_dim[2]),
        )
        self._vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long().to(self.device)

        # Convert voxel coordinates to world coordinates
        self._world_c = self._vol_origin + (self._voxel_size * self._vox_coords)
        self._world_c = torch.cat([
            self._world_c, torch.ones(len(self._world_c), 1, device=self.device)], dim=1)

        self.reset()

        # print("[*] voxel volume: {} x {} x {}".format(*self._vol_dim))
        # print("[*] num voxels: {:,}".format(self._num_voxels))

    def reset(self):
        self._tsdf_vol = torch.ones(*self._vol_dim).to(self.device)
        self._weight_vol = torch.zeros(*self._vol_dim).to(self.device)
        self._color_vol = torch.zeros(*self._vol_dim).to(self.device)

    def integrate(self, depth_im, cam_intr, cam_pose, obs_weight):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          obs_weight (float): The weight to assign to the current observation.
        """
        cam_pose = cam_pose.float().to(self.device)
        cam_intr = cam_intr.float().to(self.device)
        depth_im = depth_im.float().to(self.device)
        # print(depth_im.shape)
        im_h, im_w = depth_im.shape
        weight_vol, tsdf_vol = self._integrate_func(
            depth_im,
            cam_intr,
            cam_pose,
            obs_weight,
            self._world_c,
            self._vox_coords,
            self._weight_vol,
            self._tsdf_vol,
            self._sdf_trunc,
            im_h, im_w,
        )
        self._weight_vol = weight_vol
        self._tsdf_vol = tsdf_vol

    def get_volume(self):
        return self._tsdf_vol, self._weight_vol

    @property
    def sdf_trunc(self):
        return self._sdf_trunc

    @property
    def voxel_size(self):
        return self._voxel_size
