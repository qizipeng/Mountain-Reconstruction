import torch
import torch.nn as nn
from einops import rearrange, repeat
import torchsparse
import torchsparse.nn as spnn
from torchsparse.tensor import PointTensor
from torchsparse.utils import *
import torch.functional as F
from ctypes import *

import open3d as o3d

import time
lib=cdll.LoadLibrary('/home/qzp/selfRecon/models/thread.so')

from ops.torchsparse_utils import *
import numpy as np
__all__ = ['SPVCNN', 'SConv3d', 'ConvGRU']
def Convert1DToCArray(TYPE, ary):
    arow = TYPE(*ary.tolist())
    return arow
def Convert2DToCArray(ary):
    ROW = c_float * len(ary[0])
    rows = []
    for i in range(len(ary)):
        rows.append(Convert1DToCArray(ROW, ary[i]))
    MATRIX = ROW * len(ary)
    return MATRIX(*rows)

class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1), spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

class occ_dis(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.dropout = kwargs['dropout']

        cr = kwargs.get('cr', 1.0)
        cs = [32, 64, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            spnn.Conv3d(kwargs['in_channels'], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )
        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[2]),
                nn.BatchNorm1d(cs[2]),
                nn.ReLU(True),
            )
        ])

        # self.weight_initialization()

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # x: SparseTensor z: PointTensor
        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        z1 = voxel_to_point(x2, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)
        return z1.F

class tsdf_dis(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.dropout = kwargs['dropout']

        cr = kwargs.get('cr', 1.0)
        cs = [32, 64, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            spnn.Conv3d(kwargs['in_channels'], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )
        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[2]),
                nn.BatchNorm1d(cs[2]),
                nn.ReLU(True)
            )
        ])
        self.point_dis = nn.Sequential(
                nn.Linear(cs[2], cs[1]),
                nn.Linear(cs[1], cs[0]),
                nn.Linear(cs[0], 1),
            )
        # self.weight_initialization()

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # x: SparseTensor z: PointTensor
        x0 = initial_voxelize(z, self.pres, self.vres)
        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        z1 = voxel_to_point(x2, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)
        z1.F = self.point_dis(z1.F)
        return z1.F





class SPVCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.dropout = kwargs['dropout']

        cr = kwargs.get('cr', 1.0)
        cs = [32, 64, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            spnn.Conv3d(kwargs['in_channels'], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[2], cs[3], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[3], cs[4], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
            )
        ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[2]),
                nn.BatchNorm1d(cs[2]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[2], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            )
        ])

        # self.weight_initialization()

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         init_type = 'kaiming'
    #         init_gain = 0.02
    #         if isinstance(m, nn.Linear):
    #             if init_type == 'normal':
    #                 init.normal_(m.weight.data, 0.0, init_gain)
    #             elif init_type == 'xavier':
    #                 init.xavier_normal_(m.weight.data, gain=init_gain)
    #             elif init_type == 'kaiming':
    #                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    #             elif init_type == 'orthogonal':
    #                 init.orthogonal_(m.weight.data, gain=init_gain)
    #             else:
    #                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 init.constant_(m.bias.data, 0.0)

    def forward(self, z):
        # x: SparseTensor z: PointTensor
        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        z1 = voxel_to_point(x2, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y3 = point_to_voxel(x2, z1)
        if self.dropout:
            y3.F = self.dropout(y3.F)
        y3 = self.up1[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up1[1](y3)

        y4 = self.up2[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up2[1](y4)
        z3 = voxel_to_point(y4, z1)
        z3.F = z3.F + self.point_transforms[1](z1.F)

        return z3.F


class SConv3d(nn.Module):
    def __init__(self, inc, outc, pres, vres, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = spnn.Conv3d(inc,
                               outc,
                               kernel_size=ks,
                               dilation=dilation,
                               stride=stride)
        self.point_transforms = nn.Sequential(
            nn.Linear(inc, outc),
        )
        self.pres = pres
        self.vres = vres

    def forward(self, z):
        x = initial_voxelize(z, self.pres, self.vres)
        x = self.net(x)
        out = voxel_to_point(x, z, nearest=False)
        out.F = out.F + self.point_transforms(z.F)
        return out


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128, pres=1, vres=1):
        super(ConvGRU, self).__init__()
        self.convz = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convr = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convq = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)

    #     self.weight_initialization()
    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         init_type = 'kaiming'
    #         init_gain = 0.02
    #         if isinstance(m, nn.Linear):
    #             if init_type == 'normal':
    #                 init.normal_(m.weight.data, 0.0, init_gain)
    #             elif init_type == 'xavier':
    #                 init.xavier_normal_(m.weight.data, gain=init_gain)
    #             elif init_type == 'kaiming':
    #                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    #             elif init_type == 'orthogonal':
    #                 init.orthogonal_(m.weight.data, gain=init_gain)
    #             else:
    #                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 init.constant_(m.bias.data, 0.0)

    def forward(self, h, x):
        '''

        :param h: PintTensor
        :param x: PintTensor
        :return: h.F: Tensor (N, C)
        '''
        hx = PointTensor(torch.cat([h.F, x.F], dim=1), h.C)

        z = torch.sigmoid(self.convz(hx).F)
        r = torch.sigmoid(self.convr(hx).F)
        x.F = torch.cat([r * h.F, x.F], dim=1)
        q = torch.tanh(self.convq(x).F)

        h.F = (1 - z) * h.F + z * q
        return h.F


class Atten(nn.Module):
    def __init__(self, ch_in , hidden_dim=128, dropout = 0.3):
        super(Atten, self).__init__()
        self.ch_in = ch_in
        self.hd = hidden_dim
        self.heads = 4
        self.inner_dim = self.hd * self.heads
        self.scale = self.ch_in ** -0.5

        self.softmax = True
        self.to_q = nn.Linear(self.ch_in, self.inner_dim, bias=False)
        self.to_k = nn.Linear(self.ch_in, self.inner_dim, bias=False)
        self.to_v = nn.Linear(self.ch_in, self.inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.ch_in),
            nn.BatchNorm1d(self.hd),
            nn.ReLU(True),
            nn.Dropout(dropout)
        )

    def forward(self,h ,x): ##h: b n K+1 c x:b n c
        n_points,K,c = h.shape
        x_q = self.to_q(x) #n head*c
        x_q = x_q.reshape(n_points,self.heads,-1)
        x_q = x_q.permute(1, 0, 2).contiguous() ##head n c
        x_q = x_q.unsqueeze(-1)##head n c 1

        h_k = self.to_k(h) #n k+1 head*c
        h_k = h_k.reshape(n_points,K,self.heads,-1)
        h_k = h_k.permute(2, 0, 1,3).contiguous() ##head n k+1 c

        h_v = self.to_v(h) #n k+1 head*c
        h_v = h_v.reshape(n_points,K,self.heads,-1)
        h_v = h_v.permute(2, 0, 3,1).contiguous() ##head n c k+1

        weight = torch.matmul(h_k, x_q)  ##head n (k+1) 1
        weight = weight* self.scale

        if self.softmax:
            weight = weight.softmax(dim=-1)

        # attn = dots
        # vis_tmp(dots)

        out = torch.matmul(h_v, weight) ##head n c 1
        out = out.squeeze(-1).permute(1, 0, 2).contiguous()
        out = out.reshape(n_points, self.inner_dim)
        out = self.to_out(out)
        # vis_tmp2(out)

        return out



class transfusion(nn.Module):
    def __init__(self, ch_in , hidden_dim=128, input_dim=192 + 128, pres=1, vres=1):
        super(transfusion, self).__init__()
        self.vres = vres

        self.ch_in = ch_in
        self.hd = hidden_dim
        self.topk = 5
        self.fenkuai = [8000,5000,3000]
        self.head = 4

        self.point_transforms_h = nn.Sequential(
            nn.Linear(self.ch_in,  self.ch_in),
            nn.BatchNorm1d(self.hd),
            nn.ReLU(True),
        )
        # self.point_transforms_x = nn.Sequential(
        #     nn.Linear(self.ch_in,  self.hd),
        #     nn.BatchNorm1d(self.hd),
        #     nn.ReLU(True),
        # )
        # self.point_transforms_z = nn.Sequential(
        #     nn.Linear(self.hd, self.ch_in),
        #     nn.BatchNorm1d(self.hd),
        #     nn.ReLU(True),
        # )
        self.crossatten = Atten(self.ch_in, self.hd)
        self.localatten = Atten(self.ch_in, self.hd)
        self.convz = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convr = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convq = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)

    #     self.weight_initialization()
    #
    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         init_type = 'kaiming'
    #         init_gain = 0.02
    #         if isinstance(m, nn.Linear):
    #             if init_type == 'normal':
    #                 init.normal_(m.weight.data, 0.0, init_gain)
    #             elif init_type == 'xavier':
    #                 init.xavier_normal_(m.weight.data, gain=init_gain)
    #             elif init_type == 'kaiming':
    #                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    #             elif init_type == 'orthogonal':
    #                 init.orthogonal_(m.weight.data, gain=init_gain)
    #             else:
    #                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 init.constant_(m.bias.data, 0.0)
    def getnewpoints(self,x_f, h_f, x_c, h_c, n_points, localatten = False):
        # x_c = x_c.cpu()
        # h_c = h_c.cpu()
        index = torch.matmul(x_c,h_c)
        if x_c.shape[0] < self.topk:
            # x_f = self.point_transforms_z(x_f)
            return x_f
        # print(index.shape)
        _,index = torch.topk(index,self.topk,0)

        index = index.cuda().to("cuda:0")
        index = index.view(-1).unsqueeze(0)  # 1 n-points*7
        index = index.repeat(self.hd, 1)
        index = index.permute(1, 0).contiguous()
        index = index.type(torch.int64)  ### n-points*7 ch

        # h_f = self.point_transforms_h(h_f)
        # x_f = self.point_transforms_x(x_f)  ##n-poins c
        x_1 = x_f.unsqueeze(1)
        # x_2 = x_f.unsqueeze(-1)
        h_f = h_f.gather(dim=0, index=index)  ##n-point*k c
        h_f = h_f.view(n_points, self.topk, -1)  # n_points k c
        h_f = torch.cat([h_f,x_1],1) # n_points k+1 c
        if localatten:
            new_points = self.crossatten(h_f,x_f)  ##n k+1 c; n c
        else:
            new_points = self.localatten(h_f, x_f)  ##n k+1 c; n c


        # weight = torch.matmul(h_f, x_2)  ##n*(k+1)*1
        # weight = torch.softmax(weight, dim=1)
        #
        # h_f = h_f.permute(0, 2, 1)  ###n*c*(k+1)
        # x_f = torch.matmul(h_f, weight)  ### n*c*1
        #
        # x_f = x_f.view(n_points, -1)
        # x_f = self.point_transforms_z(x_f)

        return new_points

    def forward(self, h, x, level=None):
        NeuralRecon_fusion = True
        '''

        :param h: PintTensor
        :param x: PintTensor
        :return: h.F: Tensor (N, C)
        '''

        #print("fusion")
        ##h :global x: local
        h_f = h.F
        x_f = x.F
        x_c = x.C[:,1:]
        h_c = h.C[:,1:]
        h_c = h_c.permute(1, 0).contiguous()
        n_points = x_f.shape[0]
        self.fenkuai = int(2*n_points**(2/3))

        # print(self.fenkuai)

        if n_points<self.fenkuai:

            result = self.getnewpoints(x_f,h_f,x_c,h_c,n_points)
        else:
            result = []
            for i in range(n_points//self.fenkuai):
                result.append(
                    self.getnewpoints(x_f[i * self.fenkuai:(i + 1) * self.fenkuai, :],
                                      h_f[i * self.fenkuai:(i + 1) * self.fenkuai, :],
                                      x_c[i * self.fenkuai:(i + 1) * self.fenkuai, :],
                                      h_c[:, i * self.fenkuai:(i + 1) * self.fenkuai], n_points=self.fenkuai))
            result.append(
                self.getnewpoints(x_f[self.fenkuai*(n_points//self.fenkuai):n_points, :],
                                  h_f[self.fenkuai*(n_points//self.fenkuai):n_points, :],
                                  x_c[self.fenkuai*(n_points//self.fenkuai):n_points, :],
                                  h_c[:, self.fenkuai*(n_points//self.fenkuai):n_points], n_points= n_points-self.fenkuai*(n_points//self.fenkuai)))

            result = torch.cat(result,0)
        result = self.point_transforms_h(result)

        # x_c_t = x_c.permute(1, 0).contiguous()
        # if n_points < self.fenkuai:
        #
        #     result_2 = self.getnewpoints(result, result, x_c, x_c_t, n_points)
        # else:
        #     result_2 = []
        #     for i in range(n_points // self.fenkuai):
        #         result_2.append(
        #             self.getnewpoints(result[i * self.fenkuai:(i + 1) * self.fenkuai, :],
        #                               result[i * self.fenkuai:(i + 1) * self.fenkuai, :],
        #                               x_c[i * self.fenkuai:(i + 1) * self.fenkuai, :],
        #                               x_c_t[:, i * self.fenkuai:(i + 1) * self.fenkuai], n_points=self.fenkuai,localatten = True))
        #     result_2.append(
        #         self.getnewpoints(result[self.fenkuai * (n_points // self.fenkuai):n_points, :],
        #                           result[self.fenkuai * (n_points // self.fenkuai):n_points, :],
        #                           x_c[self.fenkuai * (n_points // self.fenkuai):n_points, :],
        #                           x_c_t[:, self.fenkuai * (n_points // self.fenkuai):n_points],
        #                           n_points=n_points - self.fenkuai * (n_points // self.fenkuai),localatten = True))
        #
        #     result_2 = torch.cat(result_2, 0)


        if NeuralRecon_fusion:
            hx = PointTensor(torch.cat([h.F, result], dim=1), h.C)
            z = torch.sigmoid(self.convz(hx).F)
            r = torch.sigmoid(self.convr(hx).F)
            x.F = torch.cat([r * h.F, result], dim=1)
            q = torch.tanh(self.convq(x).F)
            result = (1 - z) * h.F + z * q
        return result


