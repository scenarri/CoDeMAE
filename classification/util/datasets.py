# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import torchvision
import pickle
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from torch.utils import data
import numpy as np
import re


def parse_line_regex(line):
    pattern = r'^(\S+)\s+(.+?)\s+(\d+)$'
    match = re.match(pattern, line.strip())
    
    if match:
        fname = match.group(1)
        path = match.group(2)
        idx = match.group(3)
        return fname, path, idx
    else:
        raise ValueError(f"无法解析的行: {line}")
    
class NWPURESISCDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, split=None):
        self.files = []
        self.targets = []
        if train:
            with open(os.path.join(root, 'train_labels_{}.txt'.format(split)), mode='r') as f:
                infos = f.readlines()
            f.close()
        else:
            with open(os.path.join(root, 'val_labels_{}.txt'.format(split)), mode='r') as f:
                infos = f.readlines()
            f.close()
        for item in infos:
            fname, path, idx = item.strip().split()
            path = path.replace('/public/multimodal/whz/datasets/downstream/classification/NWPU-RESISC45', root)
            self.files.append(os.path.join(path , fname))
            self.targets.append(int(idx))
        self.transform = transform
        print('Creating NWPU_RESISC45 dataset with {} examples'.format(len(self.targets)))
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, i):
        img_path = self.files[i]
        img = Image.open(img_path).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        return img, self.targets[i]

class AIDDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, split=None):
        self.files = []
        self.targets = []
        if train:
            with open(os.path.join(root, 'train_labels_{}.txt'.format(split)), mode='r') as f:
                infos = f.readlines()
            f.close()
        else:
            with open(os.path.join(root, 'val_labels_{}.txt'.format(split)), mode='r') as f:
                infos = f.readlines()
            f.close()
        for item in infos:
            fname, path, idx = item.strip().split()
            path = path.replace('/public/multimodal/whz/datasets/downstream/classification/AID', root)
            self.files.append(os.path.join(path , fname))
            self.targets.append(int(idx))
        self.transform = transform
        print('Creating AID dataset with {} examples'.format(len(self.targets)))
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, i):
        img_path = self.files[i]
        img = Image.open(img_path).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        return img, self.targets[i]

class FUSARDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, split=None, seed=0):
        self.files = []
        self.targets = []
        self.label_to_target = {'Bridges': 0,
                                'Cargo': 1,
                                'CoastalLands_island': 2,
                                'Fishing': 3,
                                'LandPatches': 4,
                                'OtherShip': 5,
                                'SeaClutterWaves': 6,
                                'SeaPatches': 7,
                                'StrongFalseAlarms': 8,
                                'Tanker': 9}
        
        if train:
            with open(os.path.join(root, 'split_fewshot', f'shot_{split}-seed_{seed}.txt'), mode='r') as f:
                infos = f.readlines()
            f.close()
            for item in infos:
                # fname, path, idx = item.strip().split()
                fname, path, idx = parse_line_regex(item.strip())
                self.files.append(path)
                self.targets.append(int(idx))
        else:
            clses = os.listdir(os.path.join(root, 'Val'))
            for cls in clses:
                imgs = os.listdir(os.path.join(root, 'Val', cls))
                for i in range(len(imgs)):
                    self.files.append(os.path.join(root, 'Val', cls, imgs[i]))
                    self.targets.append(int(self.label_to_target[cls]))
            
        self.transform = transform
        print('Creating FUSAR_New dataset with {} examples'.format(len(self.targets)))
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, i):
        img_path = self.files[i]
        img = Image.open(img_path).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        return img, self.targets[i]


class MSTARDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, split=None, seed=0):
        self.files = []
        self.targets = []
        self.label_to_target = {'BMP2': 0,
                                'BTR70': 1,
                                'T72': 2,
                                'BTR60': 3,
                                '2S1': 4,
                                'BRDM2': 5,
                                'D7': 6,
                                'T62': 7,
                                'ZIL131': 8,
                                'ZSU234': 9}
        if train:
            with open(os.path.join(root, 'split_fewshot', f'shot_{split}-seed_{seed}.txt'), mode='r') as f:
                infos = f.readlines()
            f.close()
            for item in infos:
                # fname, path, idx = item.strip().split()
                fname, path, idx = parse_line_regex(item.strip())
                self.files.append(path)
                self.targets.append(int(idx))
        else:
            clses = os.listdir(os.path.join(root, 'TEST'))
            for cls in clses:
                dirs = os.listdir(os.path.join(root, 'TEST', cls))
                if os.path.isdir(os.path.join(root, 'TEST', cls, dirs[0])):
                    for dir in dirs:
                        imgs = os.listdir(os.path.join(root, 'TEST', cls, dir))
                        for i in range(len(imgs)):
                            self.files.append(os.path.join(root, 'TEST', cls, dir, imgs[i]))
                            self.targets.append(int(self.label_to_target[cls]))
                else:
                    for i in range(len(dirs)):
                        self.files.append(os.path.join(root, 'TEST', cls, dirs[i]))
                        self.targets.append(int(self.label_to_target[cls]))
        
        self.transform = transform
        print('Creating MSTAR dataset with {} examples'.format(len(self.targets)))
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, i):
        img_path = self.files[i]
        img = Image.open(img_path).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        return img, self.targets[i]


class SARACDDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, split=None, seed=0):
        self.files = []
        self.targets = []
        self.label_to_target = {'A220': 0, 'A330': 1, 'ARJ21': 2, 'Boeing737': 3, 'Boeing787': 4}
        if train:
            with open(os.path.join(root, 'split_fewshot', f'shot_{split}-seed_{seed}_train.txt'), mode='r') as f:
                infos = f.readlines()
            f.close()
            for item in infos:
                fname, path, idx = item.strip().split()
                self.files.append(path)
                self.targets.append(int(idx))
        else:
            with open(os.path.join(root, 'split_fewshot', f'shot_{split}-seed_{seed}_test.txt'), mode='r') as f:
                infos = f.readlines()
            f.close()
            for item in infos:
                fname, path, idx = item.strip().split()
                self.files.append(path)
                self.targets.append(int(idx))
        
        self.transform = transform
        print('Creating SARACD dataset with {} examples'.format(len(self.targets)))
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, i):
        img_path = self.files[i]
        img = Image.open(img_path).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        return img, self.targets[i]
    
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    
    if args.dataset == 'NWPURESISC45':
        dataset = NWPURESISCDataset(root=args.data_path, train=is_train, transform=transform, split=args.data_split)
    elif args.dataset == 'AID':
        dataset = AIDDataset(root=args.data_path, train=is_train, transform=transform, split=args.data_split)
    elif args.dataset == 'FUSAR':
        dataset = FUSARDataset(root=args.data_path, train=is_train, transform=transform, split=args.data_split, seed=args.seed)
    elif args.dataset == 'MSTAR':
        dataset = MSTARDataset(root=args.data_path, train=is_train, transform=transform, split=args.data_split)
    elif args.dataset == 'SARACD':
        dataset = SARACDDataset(root=args.data_path, train=is_train, transform=transform, split=args.data_split)
    else:
        raise NotImplementedError

    return dataset


def build_transform(is_train, args):
    if args.dataset == 'ImageNet' or args.model == 'itpn_base' or args.model == 'mars':
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif args.dataset == 'NWPURESISC45':
        mean = (0.368, 0.381, 0.3436)
        std = (0.2034, 0.1853, 0.1848)
    elif args.dataset == 'AID':
        mean = (0.3978, 0.4092, 0.3685)
        std = (0.217,  0.1945, 0.192)
    elif args.dataset == 'FUSAR' or args.dataset == 'MSTAR' or args.dataset == 'SARACD':
        mean = (0.1738, 0.1738, 0.1738)
        std = (0.2226, 0.2226, 0.2226)
    else:
        raise NotImplementedError
    
    # train transform
    if is_train:
        if args.dataset == 'NWPURESISC45' or args.dataset == 'AID':
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
            return transform
        elif args.dataset == 'FUSAR' or args.dataset == 'MSTAR' or args.dataset == 'SARACD':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop([128, 128]),
                torchvision.transforms.Resize([224, 224]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ])
            return transform
        else:
            raise NotImplementedError

    # eval transform
    if args.dataset == 'NWPURESISC45' or args.dataset == 'AID':
        t = []
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)
    elif args.dataset == 'FUSAR' or args.dataset == 'MSTAR' or args.dataset == 'SARACD':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop([128, 128]),
            torchvision.transforms.Resize([224, 224]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
        return transform
    else:
        raise NotImplementedError