import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import cv2

def map_color(path, outpath, ps):
    # READ THE DEPTH
    im_depth = cv2.imread(path)
    # print(im_depth.max(), im_depth.min())
    #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
    # im_color=cv2.applyColorMap(cv2.convertScaleAbs(im_depth,alpha=15),cv2.COLORMAP_JET)
    im_color=cv2.applyColorMap(im_depth, cv2.COLORMAP_JET)
    #convert to mat png
    im=Image.fromarray(im_color)
    #save image
    im.save(os.path.join(outpath,'color_'+os.path.basename(path)))
    # im.save(os.path.join(outpath,os.path.basename(path)))
    for p in ps:
       origin_img = cv2.imread(p)
      #  print(origin_img.max(), origin_img.min())
       img_add = cv2.addWeighted(origin_img, 0.7, im_color, 0.3, 0)
       cv2.imwrite(os.path.join(outpath, os.path.basename(p).split('.')[0]+'_'+os.path.basename(path)), img_add)
   


# def img_fuse_0(outputs, ir_0s, ys, mask_irs, mask_vis):
#     fused_imgs=[]
#     fused_masks=[]
#     for k, (output, ir_0, y, mask_ir, mask_vi) in enumerate(zip(outputs, ir_0s, ys, mask_irs, mask_vis)):
#         output = output.cpu().numpy()
#         ir_0 = ir_0.cpu().numpy()
#         y = y.cpu().numpy()
#         mask_ir = mask_ir.cpu().numpy()
#         mask_vi = mask_vi.cpu().numpy()
        
#         print(k, '  MAX in map:', output.max(), ', Min in map:', output.min())
#         value = np.unique(np.logical_or(output < 0, output > 1))
#         for v in value:
#             assert v==False
#         position = np.where(output>=0.5, False, True) #if >=0.5 take ir, <0.5 take vi

#         mask_ir[position] = 0
#         mask_vi[~position] = 0
#         fused_mask = mask_vi + mask_ir
#         fused_img = ir_0*output + y*(1-output)
#         fused_imgs.append(transforms.ToTensor()(fused_mask))
#         fused_masks.append(transforms.ToTensor()(fused_img))

#     fused_img = torch.cat(fused_imgs, dim=0)
#     fused_masks = torch.cat(fused_mask, dim=0)
#     return fused_img, fused_mask



def color(pixel):
        dict = {
            0: (0, 0, 0),
            1: (64, 0, 128),
            2: (64, 64, 0),
            3: (0, 128, 192),
            4: (0, 0, 192),
            5: (128, 128, 0),
            6: (64, 64, 128),
            7: (192, 128, 128),
            8: (192, 64, 0)
        }
        return dict[pixel]

def mask2color_save(names, masks, path): 
    for k, (name, mask) in enumerate(zip(names, masks)):
      sematic_class_in_img = torch.unique(mask[0])
      # print('--=-=-=-=-==', sematic_class_in_img)
      semantc_mask_np = mask[0].clone().detach().cpu().numpy().astype(np.int16)
      rgb1 = np.uint8(semantc_mask_np.copy())
      rgb2 = np.uint8(semantc_mask_np.copy())
      rgb3 = np.uint8(semantc_mask_np.copy())
      for i in range(len(sematic_class_in_img)):
                    t = int(sematic_class_in_img[i])
                    c = color(t)
                    rgb1[rgb1 == t] = c[0]
                    rgb2[rgb2 == t] = c[1]
                    rgb3[rgb3 == t] = c[2]
      rgb = np.concatenate(
          [rgb1[:, :, np.newaxis], rgb2[:, :, np.newaxis], rgb3[:, :, np.newaxis]], axis=2)
      # print(rgb.shape)
      img_color = Image.fromarray(np.uint8(rgb))
      img_color.save(os.path.join(path, '{}_fused.png'.format(k)))

def togpu_9(device, x1, x2, x3, x4, x5, x6, x7, x8, x9):
   x1 = x1.to(device)
   x2 = x2.to(device)
   x3 = x3.to(device)
   x4 = x4.to(device)
   x5 = x5.to(device)
   x6 = x6.to(device)
   x7 = x7.to(device)
   x8 = x8.to(device)
   x9 = x9.to(device)
   return x1, x2, x3, x4, x5, x6, x7, x8, x9

def togpu_8(device, x1, x2, x3, x4, x5, x6, x7, x8):
   x1 = x1.to(device)
   x2 = x2.to(device)
   x3 = x3.to(device)
   x4 = x4.to(device)
   x5 = x5.to(device)
   x6 = x6.to(device)
   x7 = x7.to(device)
   x8 = x8.to(device)
   return x1, x2, x3, x4, x5, x6, x7, x8

def togpu_7(device, x1, x2, x3, x4, x5, x6, x7):
   x1 = x1.to(device)
   x2 = x2.to(device)
   x3 = x3.to(device)
   x4 = x4.to(device)
   x5 = x5.to(device)
   x6 = x6.to(device)
   x7 = x7.to(device)
   return x1, x2, x3, x4, x5, x6, x7

def togpu_6(device, x1, x2, x3, x4, x5, x6):
    x1 = x1.to(device)
    x2 = x2.to(device)
    x3 = x3.to(device)
    x4 = x4.to(device)
    x5 = x5.to(device)
    x6 = x6.to(device)
    return  x1, x2, x3, x4, x5, x6

def togpu_4(device, x1, x2, x3, x4):
    x1 = x1.to(device)
    x2 = x2.to(device)
    x3 = x3.to(device)
    x4 = x4.to(device)
    return  x1, x2, x3, x4

def togpu(device, ir, vi, label, mask_ir, mask_vi, ir_0, y, cb, cr):
    ir = ir.to(device)
    vi = vi.to(device)
    label = label.to(device)
    mask_ir = mask_ir.to(device)
    mask_vi = mask_vi.to(device)
    ir_0 = ir_0.to(device)
    y = y.to(device)
    cb = cb.to(device)
    cr = cr.to(device)
    return  ir, vi, label, mask_ir, mask_vi, ir_0, y, cb, cr

def togpu_1(device, ir, vi, ir_2, y, cb, cr, ir_2_sam, y_sam):
    ir = ir.to(device)
    vi = vi.to(device)
    ir_2 = ir_2.to(device)
    y = y.to(device)
    cb = cb.to(device)
    cr = cr.to(device)
    ir_2_sam = ir_2_sam.to(device)
    y_sam = y_sam.to(device)
    return ir, vi, ir_2, y, cb, cr, ir_2_sam, y_sam

def togpu_0(device, ir, vi, ir_2, y, cb, cr):
    ir = ir.to(device)
    vi = vi.to(device)
    ir_2 = ir_2.to(device)
    y = y.to(device)
    cb = cb.to(device)
    cr = cr.to(device)
    return ir, vi, ir_2, y, cb, cr

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
            .transpose(1, 3)
            .transpose(2, 3)
    )
    return out


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
            .transpose(1, 3)
            .transpose(2, 3)
    )
    return out

#------------------------------------


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

