import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable



######by given the scale and the size of input image
######we caculate the input matrix for the weight prediction network
###### input matrix for weight prediction network
def input_matrix_wpn(inH, inW, scale, add_scale=True):
    '''
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    '''
    outH, outW = int(scale*inH), int(scale*inW)

    #### mask records which pixel is invalid, 1 valid or o invalid
    #### h_offset and w_offset caculate the offset to generate the input matrix
    scale_int = int(math.ceil(scale))
    h_offset = torch.ones(inH, scale_int, 1)
    mask_h = torch.zeros(inH,  scale_int, 1)
    w_offset = torch.ones(1, inW, scale_int)
    mask_w = torch.zeros(1, inW, scale_int)
    if add_scale:
        scale_mat = torch.zeros(1,1)
        scale_mat[0,0] = 1.0/scale
        #res_scale = scale_int - scale
        #scale_mat[0,scale_int-1]=1-res_scale
        #scale_mat[0,scale_int-2]= res_scale
        scale_mat = torch.cat([scale_mat]*(inH*inW*(scale_int**2)),0)  ###(inH*inW*scale_int**2, 4)

    ####projection  coordinate  and caculate the offset 
    h_project_coord = torch.arange(0,outH, 1).float().mul(1.0/scale)
    int_h_project_coord = torch.floor(h_project_coord)

    offset_h_coord = h_project_coord - int_h_project_coord
    int_h_project_coord = int_h_project_coord.int()

    w_project_coord = torch.arange(0, outW, 1).float().mul(1.0/scale)
    int_w_project_coord = torch.floor(w_project_coord)

    offset_w_coord = w_project_coord - int_w_project_coord
    int_w_project_coord = int_w_project_coord.int()

    ####flag for   number for current coordinate LR image
    flag = 0
    number = 0
    for i in range(outH):
        if int_h_project_coord[i] == number:
            h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], flag,  0] = 1
            flag += 1
        else:
            h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], 0, 0] = 1
            number += 1
            flag = 1

    flag = 0
    number = 0
    for i in range(outW):
        if int_w_project_coord[i] == number:
            w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], flag] = 1
            flag += 1
        else:
            w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], 0] = 1
            number += 1
            flag = 1

    ## the size is scale_int* inH* (scal_int*inW)
    h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
    ####
    mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
    mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    mask_mat = torch.sum(torch.cat((mask_h,mask_w),2),2).view(scale_int*inH,scale_int*inW)
    mask_mat = mask_mat.eq(2)
    pos_mat = pos_mat.contiguous().view(1, -1,2)
    if add_scale:
        pos_mat = torch.cat((scale_mat.view(1,-1,1), pos_mat),2)

    return pos_mat,mask_mat ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

