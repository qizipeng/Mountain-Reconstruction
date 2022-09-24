import torch

def generate_grid(n_vox, interval):
    with torch.no_grad():
        # Create voxel grid
        grid_range = [torch.arange(0, n_vox[axis], interval) for axis in range(3)]
        grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2]))  # 3 dx dy dz
        grid = grid.unsqueeze(0).cuda().float()  # 1 3 dx dy dz
        grid = grid.view(1, 3, -1)
    return grid


# n_vox= [10,10,10]
# interval = 4
# coords = generate_grid(n_vox,interval)[0]
# up_coords = []
# bs = 3
# for b in range(bs):
#     up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
# up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
#
#
# batch_ind = torch.nonzero(up_coords[:, 0] == 0).squeeze(1)
# print(batch_ind.shape)
# coords_batch = up_coords[batch_ind][:, 1:]
# print(coords_batch.shape)

# print(torch.Tensor([]).view(0, 3))
#
# a = [[1,2,3],
#     [1,1,1]]
# a = torch.Tensor(a)
#
# b = [[0,0,0],
#     [0,1,0]]
# b = torch.Tensor(b)
#
# # print((a!=0).any(-1))
# # print((b!=0).any(-1))
# #
# # print(torch.nonzero((a!=0).any(-1)|(b!=0).any(-1)))
# c = a *a
#
# import functions as F
# print(c)
# c = torch.softmax(c,0)
# print(c)

import pickle
import numpy as np
import os
# fr=open('/home/qzp/3Ddata/RawData/all_tsdf_9/fragments_train.pkl','rb')
# inf = pickle.load(fr)


# full_tsdf = np.load('/home/qzp/3Ddata/RawData/all_tsdf/object3/full_tsdf_layer0.npz',
#                     allow_pickle=True)
# full_tsdf_list = full_tsdf.f.arr_0
# full_tsdf_list[full_tsdf_list!=1]=0
# print(full_tsdf_list.sum())
# a = [[[1,2,3],
#      [4,5,6]],
#      [[7,8,9],
#       [0,0,0]]]
# a = torch.Tensor(a)
# a = a.view(1,-1)
# print(a)

# a = [[1,2,3],
#      [3,4,5]]
# b = [0,1,0]
# b = torch.Tensor(b).unsqueeze(0)
# b = b.repeat(3,1)
# b = b.permute(1, 0).contiguous().type(torch.int64)
#
#
# a = torch.Tensor(a)
# print(b)
# result = a.gather(dim = 0,index = b)
# print(result)


# def getindex(n_point,ch):
#     h = w = n_point
#     n_point = n_point ** 3
#     result = np.zeros((n_point, 7))
#     for i in range(n_point):
#         result[i][0] = i
#         result[i][1] = i + 1
#         result[i][2] = i - 1
#         result[i][3] = i + h
#         result[i][4] = i - h
#         result[i][5] = i + h * w
#         result[i][6] = i - h * w
#     result = torch.Tensor(result)
#     result = result.view(-1).unsqueeze(0)
#     result[result<0]=0
#     result[result>n_point**3-1]=n_point**3-1
#     result = result.repeat(ch,1)
#     result = result.permute(1, 0).contiguous()
#     result = result.type(torch.int64)
#     return result
#
# index = getindex(2,3)
#
# index[index < 0] = 0
# index[index > 2 ** 3 - 1] = 2 ** 3 - 1
# print(torch.max(index))
# a = torch.randn((8,3))
# result = a.gather(dim = 0,index = index)
# print(result)
#
# a = a[0:3,:]
# print(a.shape)

# import caculate
# a = caculate.ca(10,2)
# print(a)
# from ctypes import cdll
from ctypes import *
import time
lib=cdll.LoadLibrary('./thread.so')

#把numpy的二维数组打包成ctypes的标准数组形式，传送给C。但存在在C中定义需要规定列数的限制，不能为如：double **a的形式
# def Convert1DToCArray(TYPE, ary):
#     arow = TYPE(*ary.tolist())
#     return arow
# def Convert2DToCArray(ary):
#     ROW = c_float * len(ary[0])
#     rows = []
#     for i in range(len(ary)):
#         rows.append(Convert1DToCArray(ROW, ary[i]))
#     MATRIX = ROW * len(ary)
#     return MATRIX(*rows)
#
# dex = 10000
# k=2
# a = np.random.randn(dex,3)
# b = np.random.randn(dex,3)
# a = torch.Tensor(a)
# b = torch.Tensor(b)
# t = time.time()
# weight = torch.matmul(a,b.permute(1, 0).contiguous())
# weight = torch.topk(weight,2,0)
# print(time.time()-t)




# caa = Convert2DToCArray(a)
# cbb = Convert2DToCArray(b)
# brr=((c_int*k)*dex)()
# t = time.time()
# lib.ca(dex,k,caa,cbb,brr)
# print("time:",time.time()-t)
# result = np.array(brr)
# print(result)

# feat = torch.randn((100,50))
#
# a = torch.ones_like(feat[:, 0]).bool()
# print(a)

# import numpy as np
# cat_data = np.load('/home/qzp/3Ddata/RawData/all_tsdf/object6/full_tsdf_layer0.npz')
#
# print(cat_data.files)
# a = cat_data['arr_0']
# print(a.shape)
# b= a==1
# print(a.sum())
# print(b.sum())
import os
# from PIL import Image
# import numpy as np
# img = Image.open('/home/qzp/3Ddata/RawData/scans/snow_27/color/0.png')
# img_np = np.array(img)
# img_np = img_np/255
# img = Image.fromarray(img_np)
# img.show()
# import torch
# a = torch.randn(size=(2,3))
# b = torch.randn(size=(2,3))
# c = a.unsqueeze(2)
# print(c.shape)
# print(torch.cat((a,b),-1).shape)
#
# print(torch.arange(0,3))


a = [[[1,2],2] for i in range(3)]
print(a)
print(a[0][1])
print(a[1][1])
print(a[2][1])
b = [x[1] for x in a]
print(b)