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
    # �?�?loss
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
        x = torch.cat((x, out), dim=1)#���growth_rate + inChannel ��ͨ��
        return x

class Dense(nn.Module):
    def __init__(self, inchannels, nDenseblocks, nDenselayers, growth_rate):
        super(Dense, self).__init__()
        self.nDenseblocks = nDenseblocks
        self.nDenselayers = nDenselayers
        self.growth_rate = growth_rate
        #����denseblock
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

from torch.utils.data import DataLoader
from SAM import Sam, ImageEncoderViT_ad, ImageEncoderViT
from functools import partial


class ContrastiveLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.vgg_loss = PerceptualLoss_CR()
        self.vgg_loss.cuda()

    def forward(self, output, y, ir, ref, y_masks, ir_masks):
        assert y_masks.shape == ir_masks.shape
        self.vgg_loss.compute_fake_C(output)
        
        # 计算DH3
        loss_DH3 = torch.tensor(0.0).cuda()
        y_masks = torch.split(y_masks, 1, dim=1)
        for idx, vi_mask in enumerate(y_masks):
            if torch.count_nonzero(vi_mask).item() == 0:
                print("{} vi_mask is zero".format(idx))
                continue
            self.vgg_loss.compute_fake_C(output * vi_mask)
            loss_DH3 += self.vgg_loss.compute_vgg_loss(ref * vi_mask) / (self.vgg_loss.compute_vgg_loss(y * vi_mask) + 1e-7)

        # 计算DH4
        loss_DH4 = torch.tensor(0.0).cuda()
        ir_masks = torch.split(ir_masks, 1, dim=1)
        for idx, ir_mask in enumerate(ir_masks):
            if torch.count_nonzero(ir_mask).item() == 0:
                print("{} ir_mask is zero".format(idx))
                continue
            self.vgg_loss.compute_fake_C(output * ir_mask)
            loss_DH4 += self.vgg_loss.compute_vgg_loss(ref * ir_mask) / (self.vgg_loss.compute_vgg_loss(ir * ir_mask) + 1e-7)

        # 总损失为DH3和DH4的和
        loss_G = loss_DH3 + loss_DH4
        
        # 只返回总损失、DH3和DH4
        return loss_G, [loss_DH3.item(), loss_DH4.item()]

        # return loss_G, [0, 0, 0, loss_DH3.item(), loss_DH4.item()]

import numpy as np
def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


class PerceptualLoss_CR(nn.Module):
    def __init__(self):
        super(PerceptualLoss_CR, self).__init__()
        self.instancenorm1 = nn.InstanceNorm2d(512, affine=False)
        self.instancenorm2 = nn.InstanceNorm2d(256, affine=False)
        self.instancenorm3 = nn.InstanceNorm2d(128, affine=False)
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.instancenorm4 = nn.InstanceNorm2d(768, affine=False)

        # self.vgg = load_vgg16()
        # self.vgg.eval()
        # for param in self.vgg.parameters():
        #     param.requires_grad = False

        self.sam = load_sam()
        self.sam.eval()
        for param in self.sam.parameters():
            param.requires_grad = True
        
        # self.temp = 2
    
    def compute_fake_C(self, img):
        img_sam = sam_preprocess(img).cuda()
        self.img_fea1_sam, self.img_fea2_sam = self.sam(img_sam)
        # img_vgg = vgg_preprocess(img)
        # self.img_fea1, self.img_fea2, self.img_fea3 = self.vgg(img_vgg)

    def compute_vgg_loss(self, target):
        target_sam = sam_preprocess(target)
        target_fea1_sam, target_fea2_sam = self.sam(target_sam)
        # target_vgg = vgg_preprocess(target)
        # target_fea1, target_fea2, target_fea3 = self.vgg(target_vgg)
        loss = 0.0


        # loss = torch.mean((self.instancenorm1(target_fea1) - self.instancenorm1(self.img_fea1)) ** 2) +\
        #       torch.mean((self.instancenorm2(target_fea2) - self.instancenorm2(self.img_fea2)) ** 2) +\
        #       torch.mean((self.instancenorm3(target_fea3) - self.instancenorm3(self.img_fea3)) ** 2)
        loss += torch.norm(self.instancenorm4(target_fea1_sam) - self.instancenorm4(self.img_fea1_sam), p=2) + \
                torch.norm(self.instancenorm4(target_fea2_sam) - self.instancenorm4(self.img_fea2_sam), p=2)
        return loss
    
def vgg_preprocess(batch):
    tensortype = type(batch.data)
    # (r, g, b) = torch.chunk(batch, 3, dim=1)
    # batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    if batch.shape[1] == 1:
        batch = batch.repeat(1, 3, 1, 1)
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    return batch

def load_vgg16():
    vgg = Vgg16()
    vgg.load_state_dict(torch.load('./ckp/vgg16.weight'), strict=False)
    print("vgg param size = {}MB".format(count_parameters_in_MB(vgg)))
    print("******** VGG loaded ********")
    return vgg


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int = 1024):
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def preprocess_3(input: torch.Tensor, target_length) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    h, w = input.shape[-2:]
    padh = target_length - h
    padw = target_length - w
    
    # Normalize colors
    # input = (input - torch.mean(input)) / torch.std(input)

    # Pad
    # input = F.pad(input, (0, padw, 0, padh))

    return input

def sam_preprocess(batch):
    target_length = batch.shape[-1]
    target_size = get_preprocess_shape(batch.shape[-2], batch.shape[-1], target_length) # h w
    batch = F.interpolate(batch, size=target_size, mode='bilinear', align_corners=True)
    batch = preprocess_3(batch, target_length)
    
    if batch.shape[1] == 1:
        batch = batch.repeat(1, 3, 1, 1)
    # batch = F.interpolate(batch, size=(1024, 1024), mode='bilinear', align_corners=True)
    return batch

def load_sam():
    encoder_embed_dim=768
    encoder_depth=2 # transformer layer num 12 default
    encoder_num_heads=12
    encoder_global_attn_indexes=[2, 5, 8, 11]
    checkpoint='/mnt/disk3/CVPR/MM24/Ours/SAM/sam_encoder_vit_b_01ec64.pth'

    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    sam = ImageEncoderViT( # ImageEncoderViT_ad
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    # sam = Sam(
    #     image_encoder=ImageEncoderViT( # ImageEncoderViT_ad
    #         depth=encoder_depth,
    #         embed_dim=encoder_embed_dim,
    #         img_size=image_size,
    #         mlp_ratio=4,
    #         norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    #         num_heads=encoder_num_heads,
    #         patch_size=vit_patch_size,
    #         qkv_bias=True,
    #         use_rel_pos=True,
    #         global_attn_indexes=encoder_global_attn_indexes,
    #         window_size=14,
    #         out_chans=prompt_embed_dim,
    #     ),
    #     pixel_mean=[123.675, 116.28, 103.53],
    #     pixel_std=[58.395, 57.12, 57.375],
    # )

    model_dict = sam.state_dict()

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            save_dict = torch.load(f)
            save_dict_1 = { k:v for k,v in save_dict.items() if k in model_dict.keys()}
            model_dict.update(save_dict_1)
        sam.load_state_dict(model_dict)
    del save_dict
    del save_dict_1
    # print("sam param size = {}MB".format(count_parameters_in_MB(sam)))
    # print("******** SAM loaded ********")

    return sam

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        x0 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # if opt.vgg_choose != "no_maxpool":
        x1 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)

        # if opt.vgg_choose != "no_maxpool":
            # if opt.vgg_maxpooling:
        # h = F.max_pool2d(h, kernel_size=2, stride=2)

        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        # if opt.vgg_choose == "relu5_1":
        return relu5_1, x1, x0
