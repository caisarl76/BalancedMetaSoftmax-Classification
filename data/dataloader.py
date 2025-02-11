"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import json
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from data.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from custum_data.new_dataset import get_dataset

# Image statistics
RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}


# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) if key == 'iNaturalist18' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]


# Dataset
class LT_Dataset(Dataset):

    def __init__(self, root, txt, dataset, transform=None, meta=False):
        self.img_path = []
        self.targets = []
        self.transform = transform

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        # save the class frequency
        if 'train' in txt and not meta:
            if not os.path.exists('cls_freq'):
                os.makedirs('cls_freq')
            freq_path = os.path.join('cls_freq', dataset + '.json')
            self.img_num_per_cls = [0 for _ in range(max(self.targets) + 1)]
            for cls in self.targets:
                self.img_num_per_cls[cls] += 1
            with open(freq_path, 'w') as fd:
                json.dump(self.img_num_per_cls, fd)
        if dataset == 'imagenet':
            self.many_shot_idx = 390
            self.median_shot_idx = 835
            ren=1000
        elif dataset == 'places':
            self.many_shot_idx = 131
            self.median_shot_idx = 259
            ren = 365
        elif dataset == 'inat':
            self.many_shot_idx = 842
            self.median_shot_idx = 4543
            ren = 8142

        self.cls_num_list = [(int)(np.sum(np.array(self.targets) == i)) for i in range(ren)]
    def get_cls_num_list(self):
        return self.cls_num_list
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, index

class small_LT_Dataset(Dataset):

    def __init__(self, root, dataset, transform=None, meta=False):
        self.img_path = []
        self.targets = []
        self.transform = transform


        # save the class frequency
        if 'train' in txt and not meta:
            if not os.path.exists('cls_freq'):
                os.makedirs('cls_freq')
            freq_path = os.path.join('cls_freq', dataset + '.json')
            self.img_num_per_cls = [0 for _ in range(max(self.targets) + 1)]
            for cls in self.targets:
                self.img_num_per_cls[cls] += 1
            with open(freq_path, 'w') as fd:
                json.dump(self.img_num_per_cls, fd)
        if dataset == 'imagenet':
            self.many_shot_idx = 390
            self.median_shot_idx = 835
        elif dataset == 'places':
            self.many_shot_idx = 131
            self.median_shot_idx = 259
        ren = 1000 if dataset == 'imagenet' else 365

        self.cls_num_list = [(int)(np.sum(np.array(self.targets) == i)) for i in range(ren)]
    def get_cls_num_list(self):
        return self.cls_num_list
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, index
# Load datasets
def load_data(data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=2, test_open=False, shuffle=True,
              cifar_imb_ratio=None, meta=False):
    if phase == 'train_plain':
        txt_split = 'train'
    else:
        txt_split = phase
    txt = './dataset/%s/%s_LT_%s.txt' % (dataset, dataset, txt_split)
    # txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))

    print('Loading data from %s' % (txt))

    if dataset == 'iNaturalist18':
        print('===> Loading iNaturalist18 statistics')
        key = 'iNaturalist18'
    else:
        key = 'default'

    if dataset == 'CIFAR10_LT':
        print('====> CIFAR10 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR10(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
    elif dataset == 'CIFAR100_LT':
        print('====> CIFAR100 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR100(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
    elif dataset == 'CIFAR100':
        print('====> CIFAR100 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR100(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
    elif dataset in ['iNaturalist18', 'imagenet', 'places','inat']:
        rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']

        if phase not in ['train', 'val']:
            transform = get_data_transform('test', rgb_mean, rgb_std, key)
        else:
            transform = get_data_transform(phase, rgb_mean, rgb_std, key)

        print('Use data transformation:', transform)
        data_root = data_root + '/' + dataset
        set_ = LT_Dataset(data_root, txt, dataset, transform, meta)
    else:
        set_ = get_dataset(data_root=data_root, dataset=dataset, phase=phase)

    print(len(set_))

    if sampler_dic and phase == 'train' and sampler_dic.get('batch_sampler', False):
        print('Using sampler: ', sampler_dic['sampler'])
        return DataLoader(dataset=set_,
                          batch_sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                          num_workers=num_workers)

    elif sampler_dic and (phase == 'train' or meta):
        print('Using sampler: ', sampler_dic['sampler'])
        # print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                          sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                          num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
