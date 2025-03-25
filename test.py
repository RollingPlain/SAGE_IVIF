import os
import sys
import time
import logging
import argparse
import numpy as np
import utils
import torchvision
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset.dataset_test import Data
from model_sub.model_sub import Network
from tqdm import tqdm

parser = argparse.ArgumentParser("ruas")
parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoints.pt', help='location of the checkpoint')
parser.add_argument('--img_dir', type=str, default='./data/test', help='path to save results')
parser.add_argument('--save_path', type=str, default='./result/test', help='path to save results')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if args.save_path == '':
    args.save_path = os.path.join(os.path.abspath(os.path.join(args.checkpoint, '../', '../')), 'result')
    print('save_path', args.save_path)

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    test_data = Data(mode='test', img_dir = args.img_dir)
    print("test_data_size: {}".format(len(test_data)))

    test_loader = DataLoader(dataset=test_data, batch_size=1, pin_memory=True, num_workers=0, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model = Network()

    if args.checkpoint!='':
        print('loading {}'.format(args.checkpoint))
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    else:
        print('no ckp available')

    model.to(device) 

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, ascii='>='):

            names = data['name']
            exts = data['ext']
            label, ir, y, cb, cr = data['label'], data['ir'], data['y'], data['cb'], data['cr']
            
            ir, y, cb, cr = utils.togpu_4(device, ir, y, cb, cr)

            output, _ = model(y, ir)
            output_colored = utils.YCrCb2RGB(torch.cat((output, cb, cr), dim=1))

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            if args.checkpoint != '':
                
                for i, (name, ext) in enumerate(zip(names, exts)):
                    save_path = os.path.join(args.save_path, f'{name}{ext}')
                    torchvision.utils.save_image(output_colored[i:i+1], save_path)

if __name__=='__main__':
    main()
