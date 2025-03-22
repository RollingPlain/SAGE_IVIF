# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, ReLU, Conv2d, MaxPool2d, Dropout2d, AvgPool2d, AdaptiveAvgPool2d


from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        out_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, out_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Adapter(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        return self.lin2(self.act(self.lin1(x))) + shortcut

class GradientLoss(nn.Module):
    # 姊?搴?loss
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernelx = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernely = [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        self.L1loss = nn.L1Loss()

    def forward(self, x, s1, s2):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        grad_f = torch.abs(sobelx) + torch.abs(sobely)
        
        sobelx_s1 = F.conv2d(s1, self.weightx, padding=1)
        sobely_s1 = F.conv2d(s1, self.weighty, padding=1)
        grad_s1 = torch.abs(sobelx_s1) + torch.abs(sobely_s1)

        sobelx_s2 = F.conv2d(s2, self.weightx, padding=1)
        sobely_s2 = F.conv2d(s2, self.weighty, padding=1)
        grad_s2 = torch.abs(sobelx_s2) + torch.abs(sobely_s2)
        
        grad_max = torch.max(grad_s1, grad_s2)
        
        loss = self.L1loss(grad_f, grad_max)
        return loss
    

class Transition(nn.Module):
    def __init__(self, inChannel, compress_rate):
        super(Transition, self).__init__()
        self.model = nn.Sequential(
            Conv2d(inChannel, (int)(inChannel * compress_rate), 1),
            Dropout2d(0.1, True),
            BatchNorm2d((int)(inChannel * compress_rate)),
            ReLU(True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Denselayer_BC(nn.Module):
    def __init__(self, inChannel, growth_rate):
        super(Denselayer_BC, self).__init__()
        self.model = nn.Sequential(
            Conv2d(inChannel, 4 * growth_rate, 1, padding=0),
            Dropout2d(0.1, False),
            BatchNorm2d(4 * growth_rate),
            ReLU(False),
            Conv2d(4*growth_rate, growth_rate, 3, padding=1),
            Dropout2d(0.1, False),
            BatchNorm2d(growth_rate),
            ReLU(False),
        )

    def forward(self, x):
        out = self.model(x)
        x = torch.cat((x, out), dim=1)#输出growth_rate + inChannel 个通道
        return x

class Dense(nn.Module):
    def __init__(self, inchannels, nDenseblocks, nDenselayers, growth_rate):
        super(Dense, self).__init__()
        self.nDenseblocks = nDenseblocks
        self.nDenselayers = nDenselayers
        self.growth_rate = growth_rate
        #插入denseblock
        self.inchan = inchannels
        layers = []
        for j in range(self.nDenseblocks):
            for i in range(self.nDenselayers):
                layers.append(Denselayer_BC(inchannels, self.growth_rate))
                inchannels += self.growth_rate
            layers.append(Transition(inchannels, 0.5))
            inchannels = (int)(inchannels * 0.5)
        # layers.pop()
        self.denseblock = nn.Sequential(*layers)
    def forward(self, x):
        x = self.denseblock(x)
        return x
    
class Conv_h(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv_h, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, (C_in+C_out)//2, 3, 1, 1),
            nn.BatchNorm2d((C_in+C_out)//2),
            # nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d((C_in+C_out)//2, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)

class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)
    
class Conv1(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # nn.Dropout(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)

    
# nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=0)
class UpSampling(nn.Module):
    def __init__(self, C=512):
        super(UpSampling, self).__init__()
        self.Up = nn.Sequential(
            nn.Conv2d(C, C // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        up = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.Up(up)
        return x
    
    
def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int = 1024): #-> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

def loss_cal_1(epoch, label, fused_mask, fused_img, ir_0, y):
    grad = GradientLoss()
    loss_mask = F.l1_loss(fused_mask, label) + F.mse_loss(fused_mask, label) #+ self.cel(fused_mask, label.squeeze(dim=1).type(torch.long))
    loss_img = F.mse_loss(fused_img, ir_0)*5 + F.mse_loss(fused_img, y) + grad(fused_img, ir_0, y)*2
    loss = loss_mask + loss_img*5
    return loss

def img_fuse_1(outputs, ir_0s, ys, mask_irs, mask_vis):
    assert torch.max(outputs)<=1 and torch.min(outputs)>=0
    value_range = '---> MAX: {}, MIN: {}'.format(torch.max(outputs).item(), torch.min(outputs).item())
    fused_mask = torch.where(outputs>=0.5, mask_irs, mask_vis) #if >=0.5 take ir
    fused_img = outputs*ir_0s + (1-outputs)*ys # The bigger outputs,the more like ir
    spliced_img = torch.where(outputs>=0.5, ir_0s, ys)
    return fused_mask, fused_img, spliced_img, value_range


def loss_cal_2(epoch, fused_img, ir_0, y):
    # loss_img = F.mse_loss(fused_img, ir_0)*5 + F.mse_loss(fused_img, y) + self.grad(fused_img, ir_0, y)*2
    loss = F.mse_loss(fused_img, ir_0) + F.mse_loss(fused_img, y)
    return loss*5

def img_fuse_2(outputs, outputs_1, ir_0s, ys):
    assert torch.max(outputs)<=1 and torch.min(outputs)>=0
    value_range = '--->ir MAX: {}, MIN: {}'.format(torch.max(outputs).item(), torch.min(outputs).item())+'--->vi MAX: {}, MIN: {}'.format(torch.max(outputs_1).item(), torch.min(outputs_1).item())
    fused_img = outputs*ir_0s + outputs_1*ys # The bigger outputs,the more like ir
    return fused_img, value_range


