import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, ReLU, Conv2d, MaxPool2d, Dropout2d, AvgPool2d, AdaptiveAvgPool2d, LeakyReLU
from .common import MLPBlock, LayerNorm2d, Conv, GradientLoss, Dense, UpSampling

import numpy as np
import os
import cv2
import math

from .attention import SpatialTransformer

class Network(nn.Module):
    def __init__(self, mask_num=4):
        super(Network, self).__init__()

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.cel =nn.CrossEntropyLoss()
        self.grad = GradientLoss()

        self.encoder_img1 = nn.Sequential(
            Conv2d(2, 16, 3, 1, 1),
            LeakyReLU(),
            Conv2d(16, 32, 3, 1, 1),
            LeakyReLU(),
        )
        self.encoder_img2 = nn.Sequential(
            Conv2d(32, 64, 3, 1, 1),
            LeakyReLU(),
        )
        self.encoder_mask1_ir = nn.Sequential(
            Conv2d(mask_num, 16, 3, 1, 1),
            LeakyReLU(),
            Conv2d(16, 32, 3, 1, 1),
            LeakyReLU(),
        )
        self.encoder_mask2_ir = nn.Sequential(
            Conv2d(32, 64, 3, 1, 1),
            LeakyReLU(),
        )

        self.encoder_mask1_vi = nn.Sequential(
            Conv2d(mask_num, 16, 3, 1, 1),
            LeakyReLU(),
            Conv2d(16, 32, 3, 1, 1),
            LeakyReLU(),

        )
        self.encoder_mask2_vi = nn.Sequential(
            Conv2d(32, 64, 3, 1, 1),
            # MaxPool2d(2),
        )
        # self.mask_transformer1_ir = SpatialTransformer(in_channels=128, n_heads=4, d_head=32, depth=1, dropout=0., context_dim=None)
        # self.mask_transformer1_vi = SpatialTransformer(in_channels=128, n_heads=4, d_head=32, depth=1, dropout=0., context_dim=None)
        # self.img_transformer1 = SpatialTransformer(in_channels=128, n_heads=4, d_head=32, depth=1, dropout=0., context_dim=None)
        self.mask_transformer1_ir = SpatialTransformer(in_channels=64, n_heads=2, d_head=32, depth=1, dropout=0., context_dim=None)
        self.mask_transformer1_vi = SpatialTransformer(in_channels=64, n_heads=2, d_head=32, depth=1, dropout=0., context_dim=None)
        self.img_transformer1 = SpatialTransformer(in_channels=64, n_heads=2, d_head=32, depth=1, dropout=0., context_dim=None)
        
        self.middle_img = nn.Sequential(
            Conv2d(128, 64, 3, 1, 1),
            LeakyReLU(),
        )

        self.middle_mask_ir = nn.Sequential(
            Conv2d(128, 64, 3, 1, 1),
            LeakyReLU(),
        )
        self.middle_mask_vi = nn.Sequential(
            Conv2d(128, 64, 3, 1, 1),
            LeakyReLU(),
        )

        self.middle_mask = nn.Sequential(
            Conv2d(128, 64, 3, 1, 1),
            LeakyReLU(),
        )

        # self.mask_transformer2 = SpatialTransformer(in_channels=128, n_heads=4, d_head=32, depth=1, dropout=0., context_dim=None)
        # self.img_transformer2 = SpatialTransformer(in_channels=128, n_heads=4, d_head=32, depth=1, dropout=0., context_dim=None)
        self.mask_transformer2 = SpatialTransformer(in_channels=64, n_heads=2, d_head=32, depth=1, dropout=0., context_dim=None)
        self.img_transformer2 = SpatialTransformer(in_channels=64, n_heads=2, d_head=32, depth=1, dropout=0., context_dim=None)

        self.img_decoder1 = nn.Sequential(
            Conv2d(256+64, 256, 3, 1, 1),
            LeakyReLU(),
            Conv2d(256, 128, 3, 1, 1),  
            LeakyReLU(),
            Conv2d(128, 64, 3, 1, 1),
            LeakyReLU(),
        )
        self.img_decoder2 = nn.Sequential(
            Conv2d(64 + 32, 64, 3, 1, 1),
            LeakyReLU(),
            Conv2d(64, 32, 3, 1, 1),
            LeakyReLU(),            
            Conv2d(32, 16, 3, 1, 1),
            LeakyReLU(),
            Conv2d(16, 1, 3, 1, 1),
            LeakyReLU(),
            # nn.Tanh(),
        )
        self.compress_and_downsample_img1 = nn.Sequential(
            Conv2d(256, 64, 3, 1, 1),
            MaxPool2d(2),
            LeakyReLU(),
        )
        self.compress_and_downsample_img2 = nn.Sequential(
            Conv2d(256, 64, 3, 1, 1),
            MaxPool2d(2),
            LeakyReLU(),
        )
        self.compress_and_downsample_img3 = nn.Sequential(
            Conv2d(256, 64, 3, 1, 1),
            # MaxPool2d(2),
            LeakyReLU(),
        )
        self.compress_and_downsample_img4 = nn.Sequential(
            Conv2d(256, 64, 3, 1, 1),
            # MaxPool2d(2),
            LeakyReLU(),
        )
        self.downsample_vi_mask1 = nn.MaxPool2d(2)
        self.downsample_ir_mask1 = nn.MaxPool2d(2)
        self.downsample_vi_mask2 = nn.MaxPool2d(2)
        self.downsample_ir_mask2 = nn.MaxPool2d(2)
        self.compress_and_upsample = nn.Sequential(
            Conv2d(256, 64, 3, 1, 1),  
            LeakyReLU(),                                         
        )
        # 定义独立的通道压缩卷积层
        self.compress_to_32_1 = nn.Conv2d(256, 32, kernel_size=1)
        self.compress_to_32_2 = nn.Conv2d(256, 32, kernel_size=1)
        self.compress_to_32_3 = nn.Conv2d(256, 32, kernel_size=1)
        self.compress_to_32_4 = nn.Conv2d(256, 32, kernel_size=1)
        self.compress_to_32_5 = nn.Conv2d(256, 32, kernel_size=1)

        self.downsample_tf_img = nn.MaxPool2d(4)
        self.downsample_tf_img1 = nn.MaxPool2d(4)
        self.downsample_tf_vi_mask1 = nn.MaxPool2d(4)
        self.downsample_tf_vi_mask = nn.MaxPool2d(4)
        self.downsample_tf_ir_mask = nn.MaxPool2d(4)
        self.downsample_tf_ir_mask1 = nn.MaxPool2d(4)



    def process_img_mask_transformers(self, img, vi_mask, ir_mask, img1, vi_mask1, ir_mask1):
        
        img_size = img.clone()
        vi_mask_size = vi_mask.clone()
        ir_mask_size = ir_mask.clone()
        
        
        
        img = self.downsample_tf_img(img)
        img1 = self.downsample_tf_img1(img1)
        vi_mask = self.downsample_tf_vi_mask(vi_mask)
        vi_mask1 = self.downsample_tf_vi_mask1(vi_mask1)
        ir_mask = self.downsample_tf_ir_mask(ir_mask)
        ir_mask1 = self.downsample_tf_ir_mask1(ir_mask1)


        img, context_list = self.img_transformer1.forward_contextlist(img)
        # print("img:", img.shape)
        assert len(context_list) != 0

        vi_mask, _ = self.mask_transformer1_vi.forward_contextlist(vi_mask, context_list)
        ir_mask, _ = self.mask_transformer1_ir.forward_contextlist(ir_mask, context_list)
        # print("vi_maska:", vi_mask.shape)
        # print("ir_maska:", ir_mask.shape)

        img4 = img.clone()
        vi_mask4 = vi_mask.clone()
        ir_mask4 = ir_mask.clone()

        # 保存cat前的ir_mask和vi_mask
        ir_mask_cat = torch.cat((ir_mask, ir_mask1), dim=1)
        vi_mask_cat = torch.cat((vi_mask, vi_mask1), dim=1)
        # print("ir_maskb:", ir_mask_cat.shape)
        # print("vi_maskb:", vi_mask_cat.shape)

        ir_mask = self.middle_mask_ir(ir_mask_cat) + ir_mask4
        vi_mask = self.middle_mask_vi(vi_mask_cat) + vi_mask4
        # print("ir_maskc:", ir_mask.shape)
        # print("vi_maskc:", vi_mask.shape)

        ir_mask5 = ir_mask.clone()
        vi_mask5 = vi_mask.clone()
        mask = torch.cat((ir_mask, vi_mask), dim=1)
        # print("maskd:", mask.shape)

        mask = self.middle_mask(mask) + ir_mask5 + vi_mask5 + vi_mask4 + ir_mask4
        # print("maske:", mask.shape)

        img = torch.cat((img, img1), dim=1)
        # print("imga:", img.shape)

        img = self.middle_img(img) + img4
        # print("imgb:", img.shape)

        img2 = img.clone()

        mask, context_list = self.mask_transformer2.forward_contextlist(mask, context_list)
        # print("maskf:", mask.shape)
        assert len(context_list) != 0

        img, _ = self.img_transformer2.forward_contextlist(img, context_list)
        # print("imgc:", img.shape)
        # print("img:", img.shape)
        # print("img1:", img1.shape)
        # print("img2:", img2.shape)
        # print("mask:", mask.shape)

        img = torch.cat((img, img1, img2, mask), dim=1)
        # print("imgd:", img.shape)

        img = F.interpolate(img, size=(img_size.size(2), img_size.size(3)), mode='bilinear', align_corners=False)
        vi_mask = F.interpolate(vi_mask, size=(vi_mask_size.size(2), vi_mask_size.size(3)), mode='bilinear', align_corners=False)
        ir_mask = F.interpolate(ir_mask,size=(ir_mask_size.size(2), ir_mask_size.size(3)), mode='bilinear', align_corners=False)

        return img, vi_mask, ir_mask  # 返回img, vi_mask, ir_mask

    def forward(self, vi, ir, vi_mask, ir_mask):
        img = torch.cat((vi, ir), dim=1)
        # print("img:", img.shape)
        # # print("vi:", vi.shape)
        # # print("ir:", ir.shape)
        # print("vi_mask:", vi_mask.shape)
        # print("ir_mask:", ir_mask.shape)

        vi_mask = self.encoder_mask1_vi(vi_mask)
        # print("vi_mask:", vi_mask.shape)
        ir_mask = self.encoder_mask1_ir(ir_mask)
        # print("ir_mask:", ir_mask.shape)
        img = self.encoder_img1(img)
        # print("img:", img.shape)

        img0 = img.clone()
        # print("img0:", img0.shape)

        vi_mask = self.encoder_mask2_vi(vi_mask)
        ir_mask = self.encoder_mask2_ir(ir_mask)
        img = self.encoder_img2(img)
        # print("vi_mask:", vi_mask.shape)
        # print("ir_mask:", ir_mask.shape)
        # print("img:", img.shape)

        img1 = img.clone()
        vi_mask1 = vi_mask.clone()
        ir_mask1 = ir_mask.clone()
        skip_img0 = img.clone()
        # print("skip_img0:",skip_img0.shape)

        # 调用封装的函数并获取vi_mask和ir_mask
        img, vi_mask, ir_mask = self.process_img_mask_transformers(img, vi_mask, ir_mask, img1, vi_mask1, ir_mask1)
        # print("out_img:",img.shape)
        # print("out_vi_mask:",vi_mask.shape)
        # print("out_ir_mask:",ir_mask.shape)
        out_img1 = img.clone()
        skip_img1 = img.clone()
        # print("skip_img1:",skip_img1.shape)
        img = self.compress_and_downsample_img1(img)
        vi_mask = self.downsample_vi_mask1(vi_mask)
        ir_mask = self.downsample_ir_mask1(ir_mask)
        img1 = img.clone()
        vi_mask1 = vi_mask.clone()
        ir_mask1 = ir_mask.clone()
        # print("down1_img:",img.shape)
        # print("down1_vi_mask:",vi_mask.shape)
        # print("down1_ir_mask:",ir_mask.shape)
        img, vi_mask, ir_mask = self.process_img_mask_transformers(img, vi_mask, ir_mask, img1, vi_mask1, ir_mask1)
        # print("out2_img:",img.shape)
        # print("out2_vi_mask:",vi_mask.shape)
        # print("out2_ir_mask:",ir_mask.shape)
        out_img2 = img.clone()
        skip_img2 = img.clone()
        # print("skip_img2:",skip_img2.shape)
        img = self.compress_and_downsample_img2(img) 
        vi_mask = self.downsample_ir_mask2(vi_mask)
        ir_mask = self.downsample_ir_mask2(ir_mask)
        # print("down2_img:",img.shape)
        # print("down2_vi_mask:",vi_mask.shape)
        # print("down2_ir_mask:",ir_mask.shape)
        img1 = img.clone()
        vi_mask1 = vi_mask.clone()
        ir_mask1 = ir_mask.clone()
        img, vi_mask, ir_mask = self.process_img_mask_transformers(img, vi_mask, ir_mask, img1, vi_mask1, ir_mask1)
        # print("out3_img:",img.shape)
        # print("out3_vi_mask:",vi_mask.shape)
        # print("out3_ir_mask:",ir_mask.shape)
        out_img3 = img.clone()
        img = F.interpolate(img,  size=(skip_img2.size(2), skip_img2.size(3)), mode='bilinear', align_corners=True)
        # print("img:",img.shape)
        # print("skip_img2:",skip_img2.shape)
        img = img + skip_img2
        img = self.compress_and_downsample_img3(img)
        
        vi_mask = F.interpolate(vi_mask, scale_factor=2.0, mode='bilinear', align_corners=True)
        ir_mask = F.interpolate(ir_mask, scale_factor=2.0, mode='bilinear', align_corners=True)
        img1 = img.clone()
        vi_mask1 = vi_mask.clone()
        ir_mask1 = ir_mask.clone()
        # print("img1:",img1.shape)
        # print("vi_mask1:",vi_mask1.shape)
        # print("ir_mask1:",ir_mask1.shape)
        # print("up2_img:",img.shape)
        # print("up2_vi_mask:",vi_mask.shape)
        # print("up2_ir_mask:",ir_mask.shape)
        img, vi_mask, ir_mask = self.process_img_mask_transformers(img, vi_mask, ir_mask, img1, vi_mask1, ir_mask1)
        # print("out4_img:",img.shape)
        # print("out4_vi_mask:",vi_mask.shape)
        # print("out4_ir_mask:",ir_mask.shape)
        out_img4 = img.clone()
        img = F.interpolate(img, scale_factor=2.0, mode='bilinear', align_corners=True)
        img = img + skip_img1
        img = self.compress_and_downsample_img4(img)
        vi_mask = F.interpolate(vi_mask, scale_factor=2.0, mode='bilinear', align_corners=True)
        ir_mask = F.interpolate(ir_mask, scale_factor=2.0, mode='bilinear', align_corners=True)
        img1 = img.clone()
        vi_mask1 = vi_mask.clone()
        ir_mask1 = ir_mask.clone()
        # print("up1_img:",img.shape)
        # print("up1_vi_mask:",vi_mask.shape)
        # print("up1_ir_mask:",ir_mask.shape)
        img, vi_mask, ir_mask = self.process_img_mask_transformers(img, vi_mask, ir_mask, img1, vi_mask1, ir_mask1)
        # print("out5_img:",img.shape)
        # print("out5_vi_mask:",vi_mask.shape)
        # print("out5_ir_mask:",ir_mask.shape)
        out_img5 = img.clone()

        img = torch.cat((img, skip_img0), dim=1)  



        img = F.interpolate(img, size=img0.shape[2:], mode='bilinear', align_corners=True)
        # print("imge:", img.shape)

        img = self.img_decoder1(img)
        # print("imgf:", img.shape)

        img = torch.cat((img, img0), dim=1)
        # print("imgg:", img.shape)
        
        img = F.interpolate(img, size=vi.shape[2:], mode='bilinear', align_corners=True)
        # print("imgh:", img.shape)

        out = self.img_decoder2(img)
        # print("out:", out.shape)

        # 将中间输出通道压缩到32，使用独立的卷积层
        out_img1 = self.compress_to_32_1(out_img1)
        out_img2 = self.compress_to_32_2(out_img2)
        out_img3 = self.compress_to_32_3(out_img3)
        out_img4 = self.compress_to_32_4(out_img4)
        out_img5 = self.compress_to_32_5(out_img5)
        # print("out_img1:",out_img1.shape)
        # print("out_img2:",out_img2.shape)
        # print("out_img3:",out_img3.shape)
        # print("out_img4:",out_img4.shape)
        # print("out_img5:",out_img5.shape)
            # 将中间输出作为字典
        intermediate_outputs = (out_img1, out_img2, out_img3, out_img4, out_img5)

        return out, intermediate_outputs


    def loss_cal(self, output, y, ir):
        loss = self.mse(output , y ) + self.mse(output , ir )
        loss_grad = self.grad(output, y, ir) * 2

        return loss, loss_grad


