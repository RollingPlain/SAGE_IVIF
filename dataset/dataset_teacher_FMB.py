import os.path
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import cv2
import torch.nn as nn
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pickle
from tqdm import tqdm
import threading
import random
from torchvision.transforms import functional as F

class Data(Dataset):
    def __init__(self, mode, use_mask_num=20, cache_mask_num=50, crop_size=(600, 800), cache_dir=None, root_dir=None):
        self.root_dir = root_dir
        self.crop_size = crop_size
        
        # 获取文件列表并保存扩展名信息
        self.img_list = []
        self.extensions = {}
        
        for filename in os.listdir(os.path.join(self.root_dir, 'Vis')):
            name, ext = os.path.splitext(filename)
            self.img_list.append(name)
            self.extensions[name] = ext
            
        self.img_dir = root_dir

        # 确认红外图像数量与可见光图像数量一致
        assert len(os.listdir(os.path.join(self.img_dir, 'Ir'))) == len(self.img_list)

        assert mode == 'train' or mode == 'test', "dataset mode not specified"
        self.mode = mode
        if mode=='train':
            # 不使用RandomResizedCrop，我们将自定义裁剪逻辑
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
        elif mode=='test':
            self.transform = transforms.Compose([])

        
        self.cache_mask_num = cache_mask_num  # 缓存中每张图片生成的掩码数量
        self.use_mask_num = min(use_mask_num, cache_mask_num)  # 实际使用的掩码数量，不能超过缓存的数量
        self.totensor = transforms.ToTensor()
        
        # 设置缓存目录
        self.cache_dir = cache_dir if cache_dir else os.path.join(self.root_dir, 'Mask_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 初始化掩码缓存
        self.mask_cache = {}
        
        # 检查是否有缓存文件 - 注意这里使用cache_mask_num作为缓存文件名的一部分
        cache_file = os.path.join(self.cache_dir, f'mask_cache_{mode}_{cache_mask_num}.pkl')
        if os.path.exists(cache_file):
            print(f"Loading mask cache from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.mask_cache = pickle.load(f)
            print(f"Loaded masks for {len(self.mask_cache)} images (cached: {cache_mask_num}, using: {use_mask_num})")
        else:
            # 初始化SAM模型并生成所有掩码
            print(f"Initializing SAM model and generating {cache_mask_num} masks per image...")
            self._initialize_sam_and_generate_masks(cache_file)
        
        # 用于跟踪是否已经打印过全零掩码警告
        self.zero_mask_warning_printed = False
    
    def _initialize_sam_and_generate_masks(self, cache_file):
        # 初始化SAM模型
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        sam = sam_model_registry["vit_b"](checkpoint='/mnt/disk3/CVPR/SAM/segment-anything-main/sam_vit_b_01ec64.pth').to(device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=128,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
            output_mode='binary_mask',
        )
        
        # 生成所有掩码并缓存
        for idx in tqdm(range(len(self.img_list)), desc="Generating masks"):
            name_0 = self.img_list[idx]
            ext = self.extensions.get(name_0, '.png')  # 获取扩展名，默认为.png
            
            ir_path_0 = os.path.join(self.img_dir, 'Ir', name_0 + ext)
            vis_path_0 = os.path.join(self.img_dir, 'Vis', name_0 + ext)
            
            # 读取图像
            ir_img = cv2.imread(ir_path_0)
            vis_img = cv2.imread(vis_path_0)
            
            # 生成掩码
            ir_patches = mask_generator.generate(ir_img)
            ir_patches.sort(key=lambda x: x['area'], reverse=True)
            
            vis_patches = mask_generator.generate(vis_img)
            vis_patches.sort(key=lambda x: x['area'], reverse=True)
            
            # 存储掩码 - 使用cache_mask_num
            ir_masks = []
            vis_masks = []
            
            for i in range(min(self.cache_mask_num, len(ir_patches), len(vis_patches))):
                ir_masks.append(ir_patches[i]['segmentation'])
                vis_masks.append(vis_patches[i]['segmentation'])
            
            self.mask_cache[name_0] = {
                'ir_masks': ir_masks,
                'vis_masks': vis_masks
            }
            
            # 每100个样本保存一次缓存，防止中断丢失
            if (idx + 1) % 100 == 0:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.mask_cache, f)
        
        # 保存最终缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(self.mask_cache, f)
        
        print(f"Mask generation complete. Saved {self.cache_mask_num} masks per image to {cache_file}")

    def random_crop(self, image, seed, target_size):
        """
        简单的随机裁剪函数，不依赖掩码
        """
        # 设置随机种子确保一致性
        torch.manual_seed(seed)
        random.seed(seed)
        
        c, h, w = image.shape
        target_h, target_w = target_size
        
        # 随机裁剪整个图像
        if h <= target_h:
            i = 0
            crop_h = h
        else:
            i = torch.randint(0, h - target_h + 1, (1,)).item()
            crop_h = target_h
            
        if w <= target_w:
            j = 0
            crop_w = w
        else:
            j = torch.randint(0, w - target_w + 1, (1,)).item()
            crop_w = target_w
            
        cropped = image[:, i:i+crop_h, j:j+crop_w]
        
        # 如果裁剪后的大小不符合目标大小，则调整大小
        if cropped.shape[1] != target_h or cropped.shape[2] != target_w:
            cropped = F.resize(cropped, target_size)
        
        return cropped

    def segmentation_aware_random_crop(self, image, mask, seed, target_size):
        """
        在包含分割区域的边界框内进行随机裁剪，然后调整大小
        处理掩码全零的情况
        
        Args:
            image: 输入图像张量 [C, H, W]
            mask: 分割掩码张量 [H, W] 或 [1, H, W]
            seed: 随机种子
            target_size: 目标大小 (h, w)
            
        Returns:
            裁剪并调整大小后的图像
        """
        # 设置随机种子确保一致性
        torch.manual_seed(seed)
        random.seed(seed)
        
        # 确保mask是2D的
        if mask.dim() == 3 and mask.size(0) == 1:
            mask = mask.squeeze(0)
        
        # 获取图像和掩码的尺寸
        c, h, w = image.shape
        target_h, target_w = target_size
        
        # 找到掩码中非零区域的坐标
        non_zero_indices = torch.nonzero(mask > 0.5, as_tuple=False)
        
        # 如果掩码为空（全零），则进行普通的随机裁剪
        if len(non_zero_indices) == 0:
            # 只打印一次警告
            if not self.zero_mask_warning_printed:
                print("Warning: Some masks are zero, performing standard random crop")
                self.zero_mask_warning_printed = True
                
            # 使用简单的随机裁剪
            return self.random_crop(image, seed, target_size)
        else:
            # 获取掩码的边界框
            min_y, min_x = non_zero_indices.min(0)[0]
            max_y, max_x = non_zero_indices.max(0)[0]
            
            # 计算边界框的尺寸
            box_h = max_y - min_y + 1
            box_w = max_x - min_x + 1
            
            # 确保边界框至少与目标大小一样大
            # 如果边界框小于目标大小，则扩展边界框
            if box_h < target_h:
                padding = target_h - box_h
                min_y = max(0, min_y - padding // 2)
                max_y = min(h - 1, max_y + padding // 2 + padding % 2)
                box_h = max_y - min_y + 1
            
            if box_w < target_w:
                padding = target_w - box_w
                min_x = max(0, min_x - padding // 2)
                max_x = min(w - 1, max_x + padding // 2 + padding % 2)
                box_w = max_x - min_x + 1
            
            # 在边界框内随机选择裁剪起点
            if box_h > target_h:
                i = min_y + torch.randint(0, box_h - target_h + 1, (1,)).item()
            else:
                i = min_y
            
            if box_w > target_w:
                j = min_x + torch.randint(0, box_w - target_w + 1, (1,)).item()
            else:
                j = min_x
            
            # 确保裁剪区域不超出图像边界
            i = min(max(0, i), h - target_h)
            j = min(max(0, j), w - target_w)
            
            # 执行裁剪
            crop_h = min(h - i, target_h)
            crop_w = min(w - j, target_w)
            cropped = image[:, i:i+crop_h, j:j+crop_w]
        
        # 如果裁剪后的大小不符合目标大小，则调整大小
        if cropped.shape[1] != target_h or cropped.shape[2] != target_w:
            cropped = F.resize(cropped, target_size)
        
        return cropped

    def __getitem__(self, idx):
        seed = torch.random.seed()

        name_0 = self.img_list[idx]
        ext = self.extensions.get(name_0, '.png')  # 获取扩展名，默认为.png

        label = []

        label_item_path = os.path.join(self.img_dir, 'Label', name_0 + ext)
        label_mask = cv2.imread(label_item_path)
        label_mask_tensor = self.totensor(cv2.cvtColor(label_mask, cv2.COLOR_BGR2GRAY))

        ir_path_0 = os.path.join(self.img_dir, 'Ir', name_0 + ext)
        vis_path_0 = os.path.join(self.img_dir, 'Vis', name_0 + ext)
        ir_0 = cv2.imread(ir_path_0)
        vi_0 = cv2.imread(vis_path_0)
        ir_0_tensor = self.totensor(cv2.cvtColor(ir_0, cv2.COLOR_BGR2GRAY))
        vi_0_tensor = self.totensor(cv2.cvtColor(vi_0, cv2.COLOR_BGR2YCrCb)) # CHW
        
        # 在训练模式下，使用分割感知随机裁剪
        if self.mode == 'train':
            # 检查标签掩码是否全零
            if torch.sum(label_mask_tensor) > 0:
                # 先对标签掩码进行裁剪
                label_mask_tensor = self.segmentation_aware_random_crop(label_mask_tensor, label_mask_tensor, seed, self.crop_size)
                
                # 对其他图像使用相同的裁剪参数
                ir_0_tensor = self.segmentation_aware_random_crop(ir_0_tensor, label_mask_tensor, seed, self.crop_size)
                vi_0_tensor = self.segmentation_aware_random_crop(vi_0_tensor, label_mask_tensor, seed, self.crop_size)
            else:
                # 如果标签掩码全零，使用简单的随机裁剪
                if not self.zero_mask_warning_printed:
                    print("Warning: Label mask is zero, performing standard random crop")
                    self.zero_mask_warning_printed = True
                
                # 使用相同的种子进行简单的随机裁剪
                label_mask_tensor = self.random_crop(label_mask_tensor, seed, self.crop_size)
                ir_0_tensor = self.random_crop(ir_0_tensor, seed, self.crop_size)
                vi_0_tensor = self.random_crop(vi_0_tensor, seed, self.crop_size)
            
            # 应用其他变换（如水平翻转）
            torch.manual_seed(seed)
            label_mask_tensor = self.transform(label_mask_tensor)
            
            torch.manual_seed(seed)
            ir_0_tensor = self.transform(ir_0_tensor)
            
            torch.manual_seed(seed)
            vi_0_tensor = self.transform(vi_0_tensor)
        
        y_0 = vi_0_tensor[0, :, :].unsqueeze(dim=0).clone()
        cb = vi_0_tensor[1, :, :].unsqueeze(dim=0)
        cr = vi_0_tensor[2, :, :].unsqueeze(dim=0)

        irs = []
        ys = []
        
        # 从缓存中获取掩码
        cached_masks = self.mask_cache.get(name_0)
        if cached_masks:
            ir_img = cv2.imread(ir_path_0)
            vis_img = cv2.imread(vis_path_0)
            
            # 计数有效掩码
            valid_mask_count = 0
            
            # 注意这里使用use_mask_num而不是cache_mask_num
            for i in range(min(self.cache_mask_num, len(cached_masks['ir_masks']))):
                # 检查掩码是否全零
                ir_mask = cached_masks['ir_masks'][i]
                vis_mask = cached_masks['vis_masks'][i]
                
                if not np.any(ir_mask) or not np.any(vis_mask):
                    # 跳过全零掩码，但不打印警告
                    continue
                
                # 应用红外掩码
                ir_position = ~ir_mask
                ir_masked = ir_img.copy()
                ir_masked[ir_position] = 0
                
                # 应用可见光掩码
                vis_position = ~vis_mask
                vis_masked = vis_img.copy()
                vis_masked[vis_position] = 0
                
                try:
                    ir_2_tensor = self.totensor(cv2.cvtColor(ir_masked, cv2.COLOR_BGR2GRAY))
                    vi_2_tensor = self.totensor(cv2.cvtColor(vis_masked, cv2.COLOR_BGR2YCrCb))
                    
                    # 在训练模式下，使用分割感知随机裁剪
                    if self.mode == 'train':
                        # 使用与原始图像相同的裁剪和变换
                        if torch.sum(label_mask_tensor) > 0:
                            ir_2_tensor = self.segmentation_aware_random_crop(ir_2_tensor, label_mask_tensor, seed, self.crop_size)
                            vi_2_tensor = self.segmentation_aware_random_crop(vi_2_tensor, label_mask_tensor, seed, self.crop_size)
                        else:
                            ir_2_tensor = self.random_crop(ir_2_tensor, seed, self.crop_size)
                            vi_2_tensor = self.random_crop(vi_2_tensor, seed, self.crop_size)
                        
                        torch.manual_seed(seed)
                        ir_2_tensor = self.transform(ir_2_tensor)
                        
                        torch.manual_seed(seed)
                        vi_2_tensor = self.transform(vi_2_tensor)
                    
                    y = vi_2_tensor[0, :, :].unsqueeze(dim=0)
                    
                    irs.append(ir_2_tensor)
                    ys.append(y)
                    
                    # 增加有效掩码计数
                    valid_mask_count += 1
                    
                    # 如果已经收集了足够的有效掩码，就退出循环
                    if valid_mask_count >= self.use_mask_num:
                        break
                        
                except Exception as e:
                    # 出错时继续，但不打印详细错误信息
                    continue
        
        # 如果掩码数量不足，用原图填充
        while len(irs) < self.use_mask_num:
            irs.append(ir_0_tensor.clone())
            ys.append(y_0.clone())

        ys_0 = torch.cat(ys, dim=0)
        irs_0 = torch.cat(irs, dim=0)
        
        result = {'name':name_0, 'irs':irs_0, 'ys':ys_0, 'label':label, 'ir':ir_0_tensor, 'y':y_0, 'cb':cb, 'cr':cr, 'label_mask': label_mask_tensor}

        return result

    def trans(self, x, seed):
        torch.manual_seed(seed)
        x = self.transform(x)
        return x

    def __len__(self):
        return len(self.img_list)
