import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from .common import MLPBlock, LayerNorm2d, Conv, GradientLoss, Dense, UpSampling, ContrastiveLoss


class Losses(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mse = nn.MSELoss()
        self.grad = GradientLoss()
        self.contrast = ContrastiveLoss()

    def cal(self, output, y, ir, ref, y_mask, ir_mask):
        loss_fuse = 3*self.mse(output , y ) + 2*self.mse(output , ir ) + 4*self.mse(output, ref)
        # loss_fuse = torch.tensor(0.0)
        # loss_fuse = 3*self.mse(output, ref)
        loss_grad = self.grad(output, y, ir) * 9 + self.grad(output, ref, ref) * 3
        # loss_grad = 6*self.grad(output, ref, ref)
        # loss_grad =torch.tensor(0.0)
        loss_contrast, DH_value = self.contrast(output, y, ir, ref, y_mask, ir_mask)
        # loss_contrast, DH_value = torch.tensor(0.0), [torch.tensor(0.0)] * 5

        loss_contrast /= 3000
        loss = 3*loss_fuse + 3*loss_grad + loss_contrast
        # loss = loss_contrast
        return loss, loss_fuse, loss_grad, loss_contrast, DH_value


