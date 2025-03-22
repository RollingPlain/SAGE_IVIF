import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
# 将 PIL 图像转换为灰度图并进行张量化及变换
def process_input(inputs_pil, crop_size=(600, 800), seed=2, device='cuda'):
    # 变换定义
    transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(p=0.5)
    ])
    
    # 将 PIL 图像转换为灰度图
    inputs_gray = np.array(inputs_pil)
    inputs_tensor = transforms.ToTensor()(Image.fromarray(inputs_gray))  # 转为张量
    
    # # 使用变换
    # torch.random.manual_seed(seed)
    # inputs_tensor = transform(inputs_tensor)  # 对 inputs 进行变换

    # 将处理后的输入移动到指定设备
    inputs_tensor = inputs_tensor.to(device)

    return inputs_tensor

# 计算交叉熵损失
def calculate_loss(inputs, target, device='cuda'):
    # 确保 target 是整数类型的张量，并移动到指定设备
    target = target.to(device)
        # 如果 target 的形状是 [batch_size, 1, height, width]，去掉通道维度
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)  # 变成 [batch_size, height, width]

    # 计算交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 计算损失
    loss = criterion(inputs, target)
    return loss