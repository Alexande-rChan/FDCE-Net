import os
import sys

import PIL.ImageShow
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import cv2
from torchvision import transforms
import torchvision
from natsort import natsorted
from util import *

def populate_list(gt_images_path, hazy_images_path,):

    image_list_haze_index = natsorted(os.listdir(hazy_images_path))
    image_list_gt_index = natsorted(os.listdir(gt_images_path))
    image_dataset = []
    len_haze = len(image_list_haze_index)
    len_gt = len(image_list_gt_index)
    if len_haze != len_gt:
        # one gt image to many hazy image
        rep = len_haze // len_gt
        image_list_gt_index = [[_] * rep for _ in image_list_gt_index]
        image_list_gt_index = [i for j in image_list_gt_index for i in j]

    for gt, haze in zip(image_list_gt_index, image_list_haze_index):
        image_dataset.append((os.path.join(gt_images_path, gt), os.path.join(hazy_images_path,haze)))

    train_list = image_dataset

    return train_list

def att(channal):
    cv2.normalize(channal, channal, 0, 255, cv2.NORM_MINMAX)
    M = np.ones(channal.shape, np.uint8) * 255
    img_new = cv2.subtract(M, channal)
    return img_new

def process(img):
    #img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    #img = np.array(img)
    b, g, r = cv2.split(img)
    b = att(b)
    g = att(g)
    r = att(r)
    new_image = cv2.merge([r, g,b])
    return new_image

def shrink(tensor, scale):
    c, h, w = tensor.shape
    # 需要先将尺寸从 c*h*w 转换为 1*c*h*w 以便于使用 interpolate
    new_h, new_w = h // scale, w // scale
    tensor = tensor.unsqueeze(0)
    resized_tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
    # 将尺寸从 1*c*h*w 转换回 c*h*w
    return resized_tensor.squeeze(0)

class dehazing_loader(data.Dataset):
    def __init__(self, orig_images_path, hazy_images_path, mode='train', resize=None, shrink=None, random_crop=None, base_resize = 1):
        self.train_list = populate_list(orig_images_path, hazy_images_path)
        self.val_list = populate_list(orig_images_path, hazy_images_path)
        self.resize = resize
        self.random_crop = random_crop
        self.base_resize = base_resize
        self.shrink = shrink

        seed = torch.random.seed()
        torch.random.manual_seed(seed)

        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(random_crop) if random_crop else torch.nn.Identity(),
            transforms.Resize(resize) if resize else torch.nn.Identity()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),#数据分布由[0-1]改为[-1,1]

        ])
        self.trans_gt = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(random_crop) if random_crop else torch.nn.Identity(),
            transforms.Resize(resize) if resize else torch.nn.Identity()
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 数据分布由[0-1]改为[-1,1]

        ])
    def __getitem__(self, index):

        data_clean_path, data_hazy_path = self.data_list[index]
        extension = os.path.splitext(os.path.split(data_clean_path)[-1])[-1]
        img_clean = Image.open(data_clean_path).convert('RGB')
        img_hazy = Image.open(data_hazy_path).convert('RGB')

        seed = 0
        if self.random_crop != None:
            seed = torch.random.seed()
            # 如果需要随机裁剪，则需要设置随机种子
            torch.random.manual_seed(seed)
        data_clean = self.trans(img_clean)

        if self.random_crop != None:
            # 如果需要随机裁剪，则需要设置随机种子
            torch.random.manual_seed(seed)
        data_hazy = self.trans(img_hazy)


        if self.shrink != None:
            data_clean = shrink(data_clean, self.shrink)
            data_hazy = shrink(data_hazy, self.shrink)

        if self.base_resize != 1:
            data_clean = base_resize(data_clean)
            data_hazy = base_resize(data_hazy)

        # 有时需要将图片resize到base的倍数，这时候原来的尺寸需要记录下来，在经过网络处理完成后再resize回去
        return data_clean, data_hazy, extension

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    pass