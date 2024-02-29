# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import json
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_dataset(is_train, args):
    
    transform = build_transform(is_train, args)
    #transform = build_MM_MIL_transform(is_train, args)
    #root = os.path.join(args.data_path, is_train)
    #dataset = datasets.ImageFolder(root, transform=transform)
    if args.modality == 'fundus':
        dataset = Fundus_dataset(args.data_path, is_train, transform, args.nb_classes)
    elif 'OCT' in args.modality:
        dataset = OCT_dataset(args.data_path, is_train, transform, args.nb_classes)
    elif 'multi' in args.modality:
        #dataset = MM_dataset(args.data_path, is_train, transform, args.nb_classes)
        dataset = MM_MIL_dataset(args.data_path, is_train, transform, args.nb_classes)
        
    return dataset

class Fundus_dataset(Dataset):
    def __init__(self, root, is_train, transform, nb_classes):
        '''
        数据集结构:
        -root
            -images
                -20201026_4391.fundus.jpg
                 ...
            -labels
                -train
                -val
                -test
        '''
        self.root = root
        self.labels = self.read_labels(is_train)
        self.imgs = list(self.labels.keys())
        self.transform = transform
        self.class_num = nb_classes
        
    def read_labels(self, is_train):
        label_path = os.path.join(self.root, 'labels', is_train)
        labels = dict()
        
        for l in open(label_path).readlines():
            l = json.loads(l)
            labels[l['fundus']] = l['labels']
        
        return labels
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        name = self.imgs[idx]
        img = Image.open(os.path.join(self.root, 'images', name))
        #img = cv2.imread(os.path.join(self.root, 'images', name))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #img = self.transform(image=img)['image']
        img = self.transform(img)
        
        label = torch.zeros(self.class_num)
        label[self.labels[name]] = 1        
        
        return img, label


class OCT_dataset(Dataset):
    def __init__(self, root, is_train, transform, nb_classes):
        '''
        OCT数据集，一次性读取12张OCT
        数据集结构:
        -root
            -images
                -20201026_4391_008.jpg
                 ...
            -labels
                -train
                -val
                -test
        '''
        self.root = root
        self.labels = self.read_labels(is_train)
        self.imgs = list(self.labels.keys())
        self.transform = transform
        self.class_num = nb_classes
        
    def read_labels(self, is_train):
        label_path = os.path.join(self.root, 'labels', is_train)
        labels = dict()
        
        for l in open(label_path).readlines():
            l = json.loads(l)
            name = l['fundus'].split('.')[0]
            labels[name] = l['labels']
        
        return labels
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        name = self.imgs[idx]
        imgs = []
        for i in range(12):
            img = Image.open(os.path.join(self.root, 'images', name+'_'+str(i+1).zfill(3)+'.jpg'))
            img = self.transform(img)
            #img = cv2.imread(os.path.join(self.root, 'images', name+'_'+str(i+1).zfill(3)+'.jpg'))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = self.transform(image=img)['image']
            imgs.append(img)
        
        imgs = torch.stack(imgs, 0)
        label = torch.zeros(self.class_num)
        label[self.labels[name]] = 1        
        
        return imgs, label
    
class MM_dataset(Dataset):
    def __init__(self, root, is_train, transform, nb_classes):
        '''
        数据集结构:
        -root
            -images
                -20201026_4391_008.jpg
                 ...
            -labels
                -train
                -val
                -test
        '''
        self.root = root
        self.labels = self.read_labels(is_train)
        self.imgs = list(self.labels.keys())
        self.transform = transform
        self.class_num = nb_classes
        
    def read_labels(self, is_train):
        label_path = os.path.join(self.root, 'labels', is_train)
        labels = dict()
        
        for l in open(label_path).readlines():
            l = json.loads(l)
            name = l['fundus'].split('.')[0]
            labels[name] = l['labels']
        
        return labels
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        name = self.imgs[idx]
        img_OCT = []# = Image.open(os.path.join(self.root, 'images', name))
        for i in range(12):
            img = Image.open(os.path.join(self.root, 'images', name+'_'+str(i+1).zfill(3)+'.jpg'))
            if isinstance(self.transform,list):   #用mm-mil的数据增强，OCT和彩照模态增强方式不同
                img = self.transform[1](img)
            else:
                img = self.transform(img)
            img_OCT.append(img)
            
        img_fundus = Image.open(os.path.join(self.root, 'images', name+'.fundus.jpg'))
        
        img_OCT = torch.stack(img_OCT, 0)
        if isinstance(self.transform,list):
            img_fundus = self.transform[0](img_fundus)
        else:
            img_fundus = self.transform(img_fundus)
        
        label = torch.zeros(self.class_num)
        label[self.labels[name]] = 1        
        
        return (img_fundus, img_OCT), label
    
    
class MM_MIL_dataset(Dataset):
    def __init__(self, root, is_train, transform, nb_classes):
        '''
        数据集结构:
        -root
            -images
                -20201026_4391_008.jpg
                 ...
            -labels
                -train
                -val
                -test
        '''
        if is_train == 'train':
            self.fundus_samples = 6
            self.OCT_samples = 6
        else:
            self.fundus_samples = 12
            self.OCT_samples = 12
        
        self.is_train = is_train
        self.root = root
        self.labels = self.read_labels(is_train)
        self.imgs = list(self.labels.keys())
        self.transform = transform
        self.class_num = nb_classes
    
    @staticmethod
    def sort_by_name(name):
        return int(name.split('.')[0].split('_')[-1])
        
    def read_labels(self, is_train):
        label_path = os.path.join(self.root, 'labels', is_train)
        labels = dict()
        
        for l in open(label_path).readlines():
            l = json.loads(l)
            name = l['fundus'].split('.')[0]
            all_octs = l['all_octs']
            all_octs = sorted(all_octs, key=self.sort_by_name)
            
            select_octs = l['select_octs']
            other_octs = list(set(all_octs).difference(set(select_octs)))
            octs = select_octs + list(
                np.random.choice(other_octs, self.OCT_samples - len(select_octs), replace=False))
            octs = sorted(octs, key=self.sort_by_name)

            labels[name] = (l['labels'], octs)
        
        return labels
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        name = self.imgs[idx]
        
        image = Image.open(os.path.join(self.root, 'images', name+'.fundus.jpg'))
        #image = cv2.imread(os.path.join(self.root, 'images', name+'.fundus.jpg'))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #彩照训练时重复6次，测试时使用固定的12张
        img_fundus = []
        if self.is_train == 'train':
            for i in range(self.fundus_samples):
                img = self.transform(image)#['image']
                img_fundus.append(img)

        else:
            #image = cv2.imread(os.path.join(self.root, 'images', name+'.fundus.jpg'))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tmp = fundus_fov(image)
            for img in tmp:
                img = self.transform(img)#['image']
                img_fundus.append(img)
        
        img_OCT = []# = Image.open(os.path.join(self.root, 'images', name))
        
        if self.is_train == 'train':
            for n in self.labels[name][1]:
                img = Image.open(os.path.join(self.root, 'images', n))
                img = self.transform(img)
                img_OCT.append(img)
            
        else:
            for i in range(12):
                img = Image.open(os.path.join(self.root, 'images', name+'_'+str(i+1).zfill(3)+'.jpg'))
                if isinstance(self.transform,list):   #用mm-mil的数据增强，OCT和彩照模态增强方式不同
                    img = self.transform[1](img)
                else:
                    img = self.transform(img)
                img_OCT.append(img)        
        
        img_fundus = torch.stack(img_fundus, 0)
        img_OCT = torch.stack(img_OCT, 0)
        
        label = torch.zeros(self.class_num)
        label[self.labels[name][0]] = 1        
        
        return (img_fundus, img_OCT), label
    
    
def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            #color_jitter=args.color_jitter,
            #auto_augment=args.aa,
            #interpolation='bicubic',
            #re_prob=args.reprob,
            #re_mode=args.remode,
            #re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def fundus_fov(image):
    fov = []
    #Image -> cv2
    image_224 = cv2.resize(np.array(image), (224, 224))
    image = cv2.resize(np.array(image), (448, 448))
    fov.append(image_224)  # resize
    fov.append(cv2.flip(image_224.copy(), 1))
    # ten crop
    fov.append(image[:224, :224, :].copy())  # left_top
    fov.append(cv2.flip(image[:224, :224, :].copy(), 1))
    fov.append(image[224:, :224, :].copy())  # left_bottom
    fov.append(cv2.flip(image[224:, :224, :].copy(), 1))
    fov.append(image[:224, 224:, :].copy())  # right_top
    fov.append(cv2.flip(image[:224, 224:, :].copy(), 1))
    fov.append(image[224:, 224:, :].copy())  # right_bottom
    fov.append(cv2.flip(image[224:, 224:, :].copy(), 1))
    fov.append(image[112:336, 112:336, :].copy())  # right_bottom
    fov.append(cv2.flip(image[112:336, 112:336, :].copy(), 1))
    
    #cv2 -> Image
    fov = [Image.fromarray(f) for f in fov]
    return fov
'''

def fundus_fov(image):
    fov = []
    image_256 = cv2.resize(np.array(image), (256, 256))
    image = cv2.resize(np.array(image), (512, 512))
    fov.append(image_256)  # resize
    fov.append(cv2.flip(image_256.copy(), 1))
    # ten crop
    fov.append(image[:256, :256, :].copy())  # left_top
    fov.append(cv2.flip(image[:256, :256, :].copy(), 1))
    fov.append(image[256:, :256, :].copy())  # left_bottom
    fov.append(cv2.flip(image[256:, :256, :].copy(), 1))
    fov.append(image[:256, 256:, :].copy())  # right_top
    fov.append(cv2.flip(image[:256, 256:, :].copy(), 1))
    fov.append(image[256:, 256:, :].copy())  # right_bottom
    fov.append(cv2.flip(image[256:, 256:, :].copy(), 1))
    fov.append(image[128:384, 128:384, :].copy())  # right_bottom
    fov.append(cv2.flip(image[128:384, 128:384, :].copy(), 1))
    
    #cv2 -> Image
    fov = [Image.fromarray(f) for f in fov]
    return fov
'''

def build_MM_MIL_transform(is_train, args):
    transform = []
    if is_train=='train' and 'fundus' in args.modality:
        #t = []
        #t.append(transofrms.RandomHorizontalFlip())
        #t.append(transofrms.RandomRotation(degrees=90, fill=0))
        #t.append(transofrms.RandomResizedCrop(size=256, scale=(0.25,1)))
        transform.append(A.HorizontalFlip(p=0.5))
        transform.append(A.Rotate(limit=(-90,90), p=0.5, border_mode=cv2.BORDER_CONSTANT, mask_value=0))
        transform.append(A.RandomResizedCrop(height=256, width=256, scale=(0.25,1.0)))
        transform.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
        transform.append(A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5))
        transform.append(A.RandomGamma(gamma_limit=(80,120), p=0.3))
        transform.append(A.IAAAdditiveGaussianNoise(loc=0, scale=(2.5500000000000003,12.75), p=0.5))
        transform.append(A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))
        transform.append(ToTensorV2())

        return A.Compose(transform)
    
    if is_train=='train' and 'OCT' in args.modality:
        transform.append(A.HorizontalFlip(p=0.5))
        transform.append(A.Rotate(limit=(-30,30), p=0.5, border_mode=cv2.BORDER_CONSTANT, mask_value=0))
        transform.append(A.RandomResizedCrop(height=256, width=256, scale=(0.5,1.0)))
        transform.append(A.IAAAdditiveGaussianNoise(loc=0, scale=(2.5500000000000003,12.75), p=0.5))
        transform.append(A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))
        transform.append(ToTensorV2())

        return A.Compose(transform)

    if is_train=='train' and 'multi' in args.modality:
        fundus = []
        OCT = []
        fundus.append(A.HorizontalFlip(p=0.5))
        fundus.append(A.Rotate(limit=(-90,90), p=0.5, border_mode=cv2.BORDER_CONSTANT, mask_value=0))
        fundus.append(A.RandomResizedCrop(height=256, width=256, scale=(0.25,1.0)))
        fundus.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
        fundus.append(A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5))
        fundus.append(A.RandomGamma(gamma_limit=(80,120), p=0.3))
        fundus.append(A.IAAAdditiveGaussianNoise(loc=0, scale=(2.5500000000000003,12.75), p=0.5))
        fundus.append(A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))
        fundus.append(ToTensorV2())
        
        OCT.append(A.HorizontalFlip(p=0.5))
        OCT.append(A.Rotate(limit=(-30,30), p=0.5, border_mode=cv2.BORDER_CONSTANT, mask_value=0))
        OCT.append(A.RandomResizedCrop(height=256, width=256, scale=(0.5,1.0)))
        OCT.append(A.IAAAdditiveGaussianNoise(loc=0, scale=(2.5500000000000003,12.75), p=0.5))
        OCT.append(A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))
        OCT.append(ToTensorV2())
        
        transform.append(A.Compose(fundus))
        transform.append(A.Compose(OCT))
        
        return transform
    
    transform.append(A.Resize(height=256, width=256))
    transform.append(A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))
    transform.append(ToTensorV2())
    return A.Compose(transform)
        