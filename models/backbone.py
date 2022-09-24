# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torch
#
#
# def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
#     """ Asymmetric rounding to make `val` divisible by `divisor`. With default
#     bias, will round up, unless the number is no more than 10% greater than the
#     smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
#     assert 0.0 < round_up_bias < 1.0
#     new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
#     return new_val if new_val >= round_up_bias * val else new_val + divisor
#
#
# def _get_depths(alpha):
#     """ Scales tensor depths as in reference MobileNet code, prefers rouding up
#     rather than down. """
#     depths = [32, 16, 24, 40, 80, 96, 192, 320]
#     return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]
#
# class MnasMulti(nn.Module):
#
#     def __init__(self, alpha=1.0, IsNeuralRecon = True):
#         super(MnasMulti, self).__init__()
#         depths = _get_depths(alpha)
#         if alpha == 1.0:
#             MNASNet = torchvision.models.mnasnet1_0(pretrained=True, progress=True)
#         else:
#             MNASNet = torchvision.models.MNASNet(alpha=alpha)
#
#         self.conv0 = nn.Sequential(
#             MNASNet.layers._modules['0'],
#             MNASNet.layers._modules['1'],
#             MNASNet.layers._modules['2'],
#             MNASNet.layers._modules['3'],
#             MNASNet.layers._modules['4'],
#             MNASNet.layers._modules['5'],
#             MNASNet.layers._modules['6'],
#             MNASNet.layers._modules['7'],
#             MNASNet.layers._modules['8'],
#         )
#
#         self.conv1 = MNASNet.layers._modules['9']
#         self.conv2 = MNASNet.layers._modules['10']
#
#         self.out1 = nn.Conv2d(depths[4], depths[4], 1, bias=False)
#         self.out_channels = [depths[4]]
#
#         final_chs = depths[4]
#         self.inner1 = nn.Conv2d(depths[3], final_chs, 1, bias=True)
#         self.inner2 = nn.Conv2d(depths[2], final_chs, 1, bias=True)
#
#         self.out2 = nn.Conv2d(final_chs, depths[3], 3, padding=1, bias=False)
#         self.out3 = nn.Conv2d(final_chs, depths[2], 3, padding=1, bias=False)
#         self.out_channels.append(depths[3])
#         self.out_channels.append(depths[2])
#
#
#     def forward(self, x, IsNeural = True):
#         conv0 = self.conv0(x)  #B 24 128 128 segments B 1 128 128
#         conv1 = self.conv1(conv0)
#         conv2 = self.conv2(conv1)
#
#         intra_feat = conv2
#         outputs = []
#         out = self.out1(intra_feat)
#         outputs.append(out)
#
#         intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
#         out = self.out2(intra_feat)
#         outputs.append(out)
#
#         intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
#         out = self.out3(intra_feat)
#         outputs.append(out)
#
#
#         return outputs[::-1]#,loss1, loss2, loss3]
###before if no FE

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]

class Julei(nn.Module):
    def __init__(self, channel,out_channel):
        super(Julei, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel,out_channels=channel//2,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=channel//2, out_channels=out_channel, kernel_size=1)
        # self.conv3 = nn.Conv2d(in_channels=96,out_channels=9,kernel_size=1,padding=1)
        self.bn1 = nn.BatchNorm2d(channel//2)
        self.bn2 = nn.BatchNorm2d(9)
        self.relu1 = nn.LeakyReLU(0.1)
        # self.relu2 = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(0.3)
    def forward(self,conv0):
        x = self.conv1(conv0)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        x = self.softmax(x)
        return x

class Julei2(nn.Module):
    def __init__(self, channel):
        super(Julei2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel,out_channels=channel*2,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=channel*2, out_channels=channel*2, kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_channels=channel*2,out_channels=9,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channel*2)
        self.bn2 = nn.BatchNorm2d(channel*2)
        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(1)
    def forward(self,conv0):
        x = self.conv1(conv0)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.softmax(x)
        return x



class MnasMulti(nn.Module):

    def __init__(self, alpha=1.0, IsNeuralRecon = True):
        super(MnasMulti, self).__init__()
        depths = _get_depths(alpha)
        self.julei1 = Julei(80,16)
        self.julei2 = Julei(80,32)
        self.julei3 = Julei(80,64)
        self.conv_afjulei1 =  nn.Conv2d(48,80,kernel_size=1)
        # self.bn_afjulei1 = nn.BatchNorm2d(80)
        # self.relu1 = nn.ReLU(True)
        self.conv_afjulei2 = nn.Conv2d(48, 80, kernel_size=1)
        # self.bn_afjulei2 = nn.BatchNorm2d(80)
        # self.relu2 = nn.ReLU(True)
        self.conv_afjulei3 = nn.Conv2d(48, 80, kernel_size=1)
        # self.bn_afjulei3 = nn.BatchNorm2d(80)
        # self.relu3 = nn.ReLU(True)

        # self.julei = Julei2(24)
        self.juleiloss = 0
        if not IsNeuralRecon:
            self.SCALE = 4.0
            self.B1 = self.SCALE * torch.randn(1, 1, 16, 24).cuda()
            self.B1 = self.B1.expand(-1, 32*32,-1, -1)
            # self.B1 = self.B1.expand(-1, 61 * 81, -1, -1)

            self.B2 = self.SCALE * torch.randn(1, 1, 32, 24).cuda()
            self.B2 = self.B2.expand(-1, 64*64,-1, -1)

            # self.B2 = self.B2.expand(-1, 121 * 162, -1, -1)

            self.B3 = self.SCALE * torch.randn(1, 1, 64, 24).cuda()
            self.B3 = self.B3.expand(-1, 128*128,-1, -1)
            #
            # self.B3 = self.B3.expand(-1, 242 * 324, -1, -1)

            # x = torch.arange(0,32).cuda()
            # y = torch.arange(0,32).cuda()
            # x1 ,y1 = torch.meshgrid(x,y)
            # self.corrd1 = torch.cat([x1.unsqueeze(-1),y1.unsqueeze(-1)],-1).unsqueeze(0).permute(0, 3, 1, 2).contiguous() ##1 2 128 128
            #
            # x = torch.arange(0, 64).cuda()
            # y = torch.arange(0, 64).cuda()
            # x1, y1 = torch.meshgrid(x, y)
            # self.corrd2 = torch.cat([x1.unsqueeze(-1),y1.unsqueeze(-1)],-1).unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            #
            # x = torch.arange(0, 128).cuda()
            # y = torch.arange(0, 128).cuda()
            # x1, y1 = torch.meshgrid(x, y)
            # self.corrd3 = torch.cat([x1.unsqueeze(-1),y1.unsqueeze(-1)],-1).unsqueeze(0).permute(0, 3, 1, 2).contiguous()

            # self.point_transforms = nn.Sequential(
            #     nn.Linear(96, 24),
            #     # nn.BatchNorm1d(24),
            #     # nn.ReLU(True),
            # )
            self.beforconv = nn.Conv2d(in_channels=6,out_channels=3,kernel_size=1,stride=1)
        if alpha == 1.0:
            MNASNet = torchvision.models.mnasnet1_0(pretrained=True, progress=True)
        else:
            MNASNet = torchvision.models.MNASNet(alpha=alpha)

        self.conv0 = nn.Sequential(
            MNASNet.layers._modules['0'],
            MNASNet.layers._modules['1'],
            MNASNet.layers._modules['2'],
            MNASNet.layers._modules['3'],
            MNASNet.layers._modules['4'],
            MNASNet.layers._modules['5'],
            MNASNet.layers._modules['6'],
            MNASNet.layers._modules['7'],
            MNASNet.layers._modules['8'],
        )

        self.conv1 = MNASNet.layers._modules['9']
        self.conv2 = MNASNet.layers._modules['10']

        self.out1 = nn.Conv2d(depths[4], depths[4], 1, bias=False)
        self.out_channels = [depths[4]]

        final_chs = depths[4]
        self.inner1 = nn.Conv2d(depths[3], final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(depths[2], final_chs, 1, bias=True)

        self.out2 = nn.Conv2d(final_chs, depths[3], 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, depths[2], 3, padding=1, bias=False)
        self.out_channels.append(depths[3])
        self.out_channels.append(depths[2])

    def poolfeat(self,input, prob, sp_h=2, sp_w=2):

        def feat_prob_sum(feat_sum, prob_sum, shift_feat):
            feat_sum += shift_feat[:, :-1, :, :]
            prob_sum += shift_feat[:, -1:, :, :]
            return feat_sum, prob_sum

        b, _, h, w = input.shape

        h_shift_unit = 1
        w_shift_unit = 1
        p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
        feat_ = torch.cat([input, torch.ones([b, 1, h, w]).cuda()], dim=1)  # b* (n+1) *h*w
        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w),
                                 stride=(sp_h, sp_w))  # b * (n+1) * h* w
        send_to_top_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, 2 * w_shift_unit:]
        feat_sum = send_to_top_left[:, :-1, :, :].clone()
        prob_sum = send_to_top_left[:, -1:, :, :].clone()

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w),
                                 stride=(sp_h, sp_w))  # b * (n+1) * h* w
        top = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top)

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w),
                                 stride=(sp_h, sp_w))  # b * (n+1) * h* w
        top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, :-2 * w_shift_unit]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w),
                                 stride=(sp_h, sp_w))  # b * (n+1) * h* w
        left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w),
                                 stride=(sp_h, sp_w))  # b * (n+1) * h* w
        center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit,
                 w_shift_unit:-w_shift_unit]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w),
                                 stride=(sp_h, sp_w))  # b * (n+1) * h* w
        right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w),
                                 stride=(sp_h, sp_w))  # b * (n+1) * h* w
        bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, 2 * w_shift_unit:]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w),
                                 stride=(sp_h, sp_w))  # b * (n+1) * h* w
        bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

        prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w),
                                 stride=(sp_h, sp_w))  # b * (n+1) * h* w
        bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
        feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)

        pooled_feat = feat_sum / (prob_sum + 1e-8)

        return pooled_feat

    def upfeat(self, input, prob, up_h=2, up_w=2):
        # input b*n*H*W  downsampled
        # prob b*9*h*w
        b, c, h, w = input.shape

        h_shift = 1
        w_shift = 1

        p2d = (w_shift, w_shift, h_shift, h_shift)
        feat_pd = F.pad(input, p2d, mode='constant', value=0)

        gt_frm_top_left = F.interpolate(feat_pd[:, :, :-2 * h_shift, :-2 * w_shift], size=(h * up_h, w * up_w),
                                        mode='nearest')
        feat_sum = gt_frm_top_left * prob.narrow(1, 0, 1)

        top = F.interpolate(feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
        feat_sum += top * prob.narrow(1, 1, 1)

        top_right = F.interpolate(feat_pd[:, :, :-2 * h_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
        feat_sum += top_right * prob.narrow(1, 2, 1)

        left = F.interpolate(feat_pd[:, :, h_shift:-w_shift, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
        feat_sum += left * prob.narrow(1, 3, 1)

        center = F.interpolate(input, (h * up_h, w * up_w), mode='nearest')
        feat_sum += center * prob.narrow(1, 4, 1)

        right = F.interpolate(feat_pd[:, :, h_shift:-w_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
        feat_sum += right * prob.narrow(1, 5, 1)

        bottom_left = F.interpolate(feat_pd[:, :, 2 * h_shift:, :-2 * w_shift], size=(h * up_h, w * up_w),
                                    mode='nearest')
        feat_sum += bottom_left * prob.narrow(1, 6, 1)

        bottom = F.interpolate(feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
        feat_sum += bottom * prob.narrow(1, 7, 1)

        bottom_right = F.interpolate(feat_pd[:, :, 2 * h_shift:, 2 * w_shift:], size=(h * up_h, w * up_w),
                                     mode='nearest')
        feat_sum += bottom_right * prob.narrow(1, 8, 1)

        return feat_sum
    def compute_semantic_pos_loss(self,prob_in, labxy_feat, pos_weight=0.003, kernel_size=8):
        # this wrt the slic paper who used sqrt of (mse)

        # rgbxy1_feat: B*50+2*H*W
        # output : B*9*H*w
        # NOTE: this loss is only designed for one level structure

        # todo: currently we assume the downsize scale in x,y direction are always same
        S = kernel_size
        m = pos_weight
        prob = prob_in.clone()
        kernel_size = 4
        b, c, h, w = labxy_feat.shape
        pooled_labxy = self.poolfeat(labxy_feat, prob, kernel_size, kernel_size)
        reconstr_feat = self.upfeat(pooled_labxy, prob, kernel_size, kernel_size)
        loss_map = reconstr_feat[:, -2:, :, :] - labxy_feat[:, -2:, :, :]
        # loss_map = pooled_labxy[:, -2:, :, :] - labxy_feat[:, -2:, :, :]
        # map = torch.isnan(reconstr_feat)
        # print(torch.sum(map))
        # self def cross entropy  -- the official one combined softmax
        # logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
        #
        # # map = torch.isnan(logit)
        # # print(torch.sum(map))
        #
        # loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b

        loss_sem = F.mse_loss(reconstr_feat,labxy_feat)

        loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

        # empirically we find timing 0.005 tend to better performance
        loss_sum = 0.1 * (loss_sem + loss_pos)
        loss_sem_sum = 0.5 * loss_sem
        loss_pos_sum = 0.5 * loss_pos

        return loss_sum, loss_sem_sum, loss_pos_sum, reconstr_feat

    def forward(self, x, IsNeural = True):
        conv0 = self.conv0(x)  #B 24 128 128 segments B 1 128 128
        # print(x.shape, conv0.shape)
        # if not IsNeural:
        #     corrd = self.corrd3.expand(conv0.shape[0], -1,-1, -1) #b 2 128 128
        #     pred = self.julei(conv0)
        #     conv0_XY = torch.cat([conv0,corrd],1) # b 24+2 128 128
        #     loss1,_,_,conv_new = self.compute_semantic_pos_loss(pred, conv0_XY.detach())
        #     conv0 = conv_new[:,:-2,:,:]
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = []
        out = self.out1(intra_feat)
        # print(out.shape)
        if not IsNeural:
            # corrd = self.corrd1.expand(out.shape[0], -1,-1, -1) #b 2 128 128
            # pred = self.julei1(out)
            # conv0_XY = torch.cat([out,corrd],1) # b 24+2 128 128
            # loss1,_,_,conv_new = self.compute_semantic_pos_loss(pred, conv0_XY.detach())
            # out = conv_new[:,:-2,:,:]
            #
            # out = self.conv_afjulei1(out)
            # out = self.bn_afjulei1(out)
            # out = self.relu1(out)
            # print(out.shape)
            pred = self.julei1(out)  ##b 16 32 32
            # print(pred.shape)
            pred = pred.reshape(pred.shape[0], pred.shape[1], pred.shape[2]*pred.shape[3]).permute(0, 2, 1).contiguous().unsqueeze(2) ##b n 1 16
            rff_input = torch.cat([torch.matmul(torch.sin(2*3.14*pred),self.B1),torch.matmul(torch.cos(2*3.14*pred),self.B1)],-1)##b n 1 96
            rff_input = rff_input.permute(0, 3,1, 2).contiguous().squeeze(-1).reshape(out.shape[0],-1,out.shape[2],out.shape[3]) ##b 96 128,128
            # print(rff_input.shape)
            out_result = self.conv_afjulei1(rff_input)
            out = out+out_result

        outputs.append(out)
        # intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)


        tmp1 = self.inner1(conv1)
        # print(tmp1.shape, F.interpolate(intra_feat, size = tmp1.shape[2:4], mode = 'bilinear').shape)
        intra_feat = F.interpolate(intra_feat, size = tmp1.shape[2:4], mode = 'bilinear') + tmp1

        # print(intra_feat.shape)
        if not IsNeural:
            # corrd = self.corrd2.expand(intra_feat.shape[0], -1,-1, -1) #b 2 128 128
            # pred = self.julei2(intra_feat)
            # conv0_XY = torch.cat([intra_feat,corrd],1) # b 24+2 128 128
            # loss2,_,_,conv_new = self.compute_semantic_pos_loss(pred, conv0_XY.detach())
            # intra_feat = conv_new[:,:-2,:,:]
            #
            # intra_feat = self.conv_afjulei2(intra_feat)
            # intra_feat = self.bn_afjulei2(intra_feat)
            # intra_feat = self.relu2(intra_feat)


            pred = self.julei2(intra_feat)
            # print(pred.shape)
            pred = pred.reshape(pred.shape[0], pred.shape[1], pred.shape[2]*intra_feat.shape[3]).permute(0, 2, 1).contiguous().unsqueeze(2)  ##b n 1 16
            rff_input = torch.cat([torch.matmul(torch.sin(2*3.14*pred),self.B2),torch.matmul(torch.cos(2*3.14*pred),self.B2)],-1)
            rff_input = rff_input.permute(0, 3, 1, 2).contiguous().squeeze(-1).reshape(intra_feat.shape[0], -1, intra_feat.shape[2],
                                                                                       intra_feat.shape[3])
            out_result = self.conv_afjulei2(rff_input)
            intra_feat = intra_feat+out_result
        out = self.out2(intra_feat)
        outputs.append(out)

        # intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)

        tmp2 = self.inner2(conv0)
        # print(tmp1.shape, F.interpolate(intra_feat, size = tmp1.shape[2:4], mode = 'bilinear').shape)
        intra_feat = F.interpolate(intra_feat, size = tmp2.shape[2:4], mode = 'bilinear') + tmp2

        # print(intra_feat.shape)
        if not IsNeural:
            # corrd = self.corrd3.expand(intra_feat.shape[0], -1,-1, -1) #b 2 128 128
            # pred = self.julei2(intra_feat)
            # conv0_XY = torch.cat([intra_feat,corrd],1) # b 24+2 128 128
            # loss3,_,_,conv_new = self.compute_semantic_pos_loss(pred, conv0_XY.detach())
            # intra_feat = conv_new[:,:-2,:,:]
            #
            # intra_feat = self.conv_afjulei3(intra_feat)
            # intra_feat = self.bn_afjulei3(intra_feat)
            # intra_feat = self.relu3(intra_feat)

            pred = self.julei3(intra_feat)
            # print(pred.shape)
            pred = pred.reshape(pred.shape[0], pred.shape[1],
                                pred.shape[2] * pred.shape[3]).permute(0, 2, 1).contiguous().unsqueeze(2)
            rff_input = torch.cat([torch.matmul(torch.sin(2*3.14*pred),self.B3),torch.matmul(torch.cos(2*3.14*pred),self.B3)],-1)

            rff_input = rff_input.permute(0, 3, 1, 2).contiguous().squeeze(-1).reshape(intra_feat.shape[0], -1,
                                                                                       intra_feat.shape[2],
                                                                                       intra_feat.shape[3])

            out_result = self.conv_afjulei2(rff_input)
            intra_feat = intra_feat+out_result

        out = self.out3(intra_feat)
        outputs.append(out)


        return outputs[::-1]#,loss1, loss2, loss3]
