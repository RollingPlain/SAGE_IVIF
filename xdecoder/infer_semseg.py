import os
import sys
import logging
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from xdecoder.utils.arguments import load_opt_command
from xdecoder.modeling.BaseModel import BaseModel
from xdecoder.modeling import build_model
from xdecoder.utils.visualizer import Visualizer

logger = logging.getLogger(__name__)

def process_image(image, model, transform):
    # 不再创建文件夹，因为不需要输出路径
    with torch.no_grad():
        width, height = image.size
        image_transformed = transform(image)
        image_np = np.asarray(image_transformed)
        image_ori_np = np.asarray(image)
        images = torch.from_numpy(image_np.copy()).permute(2, 0, 1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = model.forward(batch_inputs)
        # visual = Visualizer(image_ori_np)

        # sem_seg = outputs[-1]['sem_seg'].max(0)[1]
        # # demo = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5)  # rgb Image

        # sem_seg_np = sem_seg.cpu().numpy().astype(np.uint8)
        # mask_image = Image.fromarray(sem_seg_np)

        # return mask_image
                # 将输出转为适用于 nn.CrossEntropyLoss 的格式
        logits = outputs[-1]['sem_seg']  # logits 形状：[1, num_classes, height, width]
        
        return logits  # 返回 logits，而不是灰度 mask

def set_sys_args():
    sys.argv = [
        'infer_semseg.py',
        'evaluate',
        '--conf_files', '/mnt/disk3/CVPR/SAM/X-Decoder-2.0/configs/xdecoder/xdecoder_focall_lang.yaml',
        '--overrides', 'RESUME_FROM', '/mnt/disk3/CVPR/MM24/Teacher2/xdecoder/xdecoder_focall_last.pt'
    ]

def segment(image: Image.Image, model=None, transform=None):
    set_sys_args()  # 设置 sys.argv 参数

    opt, cmdline_args = load_opt_command([])  # 不传递 args

    # 强制使用 GPU 0
    opt['local_rank'] = 0 

    # 设置 device 为 cuda 或 cpu
    opt['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    local_tokenizer_path = '/mnt/disk3/CVPR/MM24/Teacher2/xdecoder/local_clip_model'
    if 'MODEL' not in opt:
        opt['MODEL'] = {}
    if 'TEXT' not in opt['MODEL']:
        opt['MODEL']['TEXT'] = {}
    opt['MODEL']['TEXT']['TOKENIZER'] = 'clip'
    opt['MODEL']['TEXT']['PRETRAINED_TOKENIZER'] = local_tokenizer_path

    pretrained_pth = '/mnt/disk3/CVPR/MM24/Teacher2/xdecoder/xdecoder_focall_last.pt'
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().to(opt['device'])

    t = [transforms.Resize(512, interpolation=Image.BICUBIC)]
    transform = transforms.Compose(t) if transform is None else transform

    # 设置模型参数
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(["road", "sidewalk", "building", "lamp", "sign", "vegetation", "sky", "person", "car", "truck", "bus", "motocycle", "bicycle", "pole", "background"], is_eval=True)
    model.model.sem_seg_head.num_classes = 14  # 固定类别数，避免使用metadata

    # 调用process_image函数处理图像并返回结果
    logits = process_image(image, model, transform)

    return logits
