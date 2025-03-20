import os.path
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import cv2



class Data(Dataset):
    def __init__(self, mode, crop_size=(600, 800), img_dir='/mnt/disk1/IVIF/M3FD_4200'):
        self.img_dir = img_dir
        
        # 获取IR和Vis文件夹中的所有文件，不限制扩展名
        ir_files = os.listdir(os.path.join(self.img_dir, 'Ir'))
        vis_files = os.listdir(os.path.join(self.img_dir, 'Vis'))
        
        # 创建文件名到扩展名的映射
        self.ir_extensions = {os.path.splitext(f)[0]: os.path.splitext(f)[1] for f in ir_files}
        self.vis_extensions = {os.path.splitext(f)[0]: os.path.splitext(f)[1] for f in vis_files}
        
        # 获取共有的基础文件名（不含扩展名）
        ir_names = set(self.ir_extensions.keys())
        vis_names = set(self.vis_extensions.keys())
        common_names = ir_names.intersection(vis_names)
        
        self.img_list = list(common_names)
        
        assert len(ir_files) >= len(self.img_list), "IR文件夹中的文件数量不足"
        assert len(vis_files) >= len(self.img_list), "Vis文件夹中的文件数量不足"

        assert mode == 'train' or mode == 'test', "dataset mode not specified"
        self.mode = mode
        if mode=='train':
            self.transform = transforms.Compose([transforms.RandomResizedCrop(crop_size), transforms.RandomHorizontalFlip(p=0.5)])
        elif mode=='test':
            self.transform = transforms.Compose([])

 
        self.totensor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        seed = torch.random.seed()

        name_0 = self.img_list[idx]

        # 使用对应的扩展名构建完整文件路径
        ir_ext = self.ir_extensions.get(name_0, '.png')  # 默认使用.png作为后备
        vis_ext = self.vis_extensions.get(name_0, '.png')
        
        # 优先使用可见光图像的扩展名
        file_ext = vis_ext
        
        ir_path_0 = os.path.join(self.img_dir, 'Ir', name_0 + ir_ext)
        vis_path_0 = os.path.join(self.img_dir, 'Vis', name_0 + vis_ext)
        
        ir_0 = cv2.imread(ir_path_0)
        vi_0 = cv2.imread(vis_path_0)

        ir_0 = self.trans(self.totensor(cv2.cvtColor(ir_0, cv2.COLOR_BGR2GRAY)), seed)
        vi_0 = self.trans(self.totensor(cv2.cvtColor(vi_0, cv2.COLOR_BGR2YCrCb)), seed)  # CHW
        y_0 = vi_0[0, :, :].unsqueeze(dim=0).clone()
        cb = vi_0[1, :, :].unsqueeze(dim=0)
        cr = vi_0[2, :, :].unsqueeze(dim=0)
        
        label = []
        
        result = {
            'name': name_0, 
            'ext': file_ext,  
            'label': label, 
            'ir': ir_0, 
            'y': y_0, 
            'cb': cb, 
            'cr': cr
                            }

        return result

    def trans(self, x, seed):
        torch.random.manual_seed(seed)
        x = self.transform(x)
        return x

    def __len__(self):
        return len(self.img_list)