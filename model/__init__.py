import os
import math 
from importlib import import_module

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel as P
import torch.nn.functional as F
from model.common import input_matrix_wpn

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.args = args
        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.fold = args.fold
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half': self.model.half()

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self, 'set_scale'):
            self.set_scale(idx_scale)
        
        if self.training: 
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)

        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            elif self.fold: 
                forward_function = self.forward_fold
            else:
                forward_function = self.model.forward

            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            return self.forward_chop(x)
        elif self.fold and not self.training: 
            return self.forward_fold(x, block_size=self.args.block_size)
        else:
            self.model.set_scale(idx_scale)
            return self.model(x)


    def save(self, apath, epoch, is_best=False):
        torch.save(
            self.state_dict(),
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                self.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )

        if self.save_models:
            torch.save(
                self.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.model.load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=True
                )
                print('load_model_mode=1')
        else:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )
            print('load_model_mode=2')

    def forward_chop(self, x, shave=10, min_size=160000):
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output


    def forward_fold(self, x, shave=10, block_size=128): 
        scale = self.scale[self.idx_scale]
        self.model.set_scale(self.idx_scale)
        n_GPUs = self.n_GPUs
        bs = self.args.batch_size
        # height, width
        b, c, h, w = x.size()
        
        # image to tokens 
        stride = block_size - shave
        ph = math.ceil((h-block_size)/stride) * stride - (h-block_size)
        pw = math.ceil((w-block_size)/stride) * stride - (w-block_size)
        th = (h + ph - block_size) // stride + 1
        tw = (w + pw - block_size) // stride + 1 
        x2 = F.pad(x, pad=(0, pw, 0, ph), mode='replicate')
        inputs = F.unfold(x2, kernel_size=block_size, stride=stride, padding=0, dilation=1)
        
        inputs = inputs.permute(0, 2, 1).contiguous()
        inputs = inputs.view(b*th*tw, 3, block_size, block_size)

        
        # inference 
        y_chops = []
        for i in range(0, inputs.size(0), bs):
            y = inputs[i:(i + bs), ...]
            y = P.data_parallel(self.model, y, range(n_GPUs))
            y_chops.append(y)
        y = torch.cat(y_chops, dim=0)

        # tokens to image 
        y = y.view(b, th*tw, 3*(block_size*scale)**2).permute(0, 2, 1).contiguous()
        h1, w1 = (th-1)*stride*scale+block_size*scale, (tw-1)*stride*scale+block_size*scale
        y = F.fold(y, output_size=(h1, w1),  kernel_size=block_size*scale, stride=stride*scale)
        norm_map = F.fold(
            F.unfold(torch.ones(1, 1, h1, w1).type(torch.FloatTensor), kernel_size=block_size*scale, stride=stride*scale), 
            output_size=(h1, w1),  kernel_size=block_size*scale, stride=stride*scale) + 1e-12 
        y /= norm_map.to(y.device)
        y = y[:, :, :h*scale, :w*scale]

        return y 
