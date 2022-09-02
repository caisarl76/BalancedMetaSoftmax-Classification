import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.randaugment import rand_augment_transform


class DTD(Dataset):
    def __init__(self, root, train=True, transform=None, rand_number=0, imb_factor=0.1, imb_type='exp',
                 random_seed=False, classwise_aug=False, alpha=0.5, magnitude=10, step_type=False):
        np.random.seed(rand_number)
        self.random_seed = random_seed
        self.classwise_aug = classwise_aug
        self.magnitude = magnitude

        self.alpha = alpha
        self.step_type = step_type
        self.step = None
        self.root_dir = os.path.join(os.getcwd(), root, 'dtd')
        self.categories = sorted(os.listdir(os.path.join(self.root_dir, 'images')))
        if train:
            excel_file = [os.path.join(self.root_dir, 'labels', 'train1.txt')]
            excel_file += [os.path.join(self.root_dir, 'labels', 'val1.txt')]
        else:
            excel_file = os.path.join(self.root_dir, 'labels', 'test1.txt')
        self.samples = []
        if isinstance(excel_file, list):
            for file in excel_file:
                self.samples += list(pd.read_csv(file)['PATH'])
        else:
            self.samples = list(pd.read_csv(excel_file)['PATH'])

        self.transform = transform
        self.targets = []
        for s in self.samples:
            class_name = s.split('/')[0]
            self.targets.append(self.categories.index(class_name))

        num_in_class = []
        for class_idx in np.unique(self.targets):
            num_in_class.append(len(np.where(self.targets == class_idx)[0]))
        self.num_in_class = num_in_class

        self.classes = np.unique(self.targets)
        self.cls_num = len(self.classes)

        if train:
            num_weak_aug, num_strong_aug = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(num_weak_aug, num_strong_aug)
        self.ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x))
                                                                               for x in [0.5273, 0.4702, 0.4235]]), )
        self.cls_num_list = self.get_cls_num_list()
        self.factor = np.array(self.cls_num_list) / self.cls_num_list[0]

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.samples) / cls_num
        num_weak_aug = []
        num_strong_aug = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = (int)(img_max * (imb_factor ** (cls_idx / (cls_num - 1.0))))
                num_weak_aug.append(num)
                else_num = (int)(img_max) - num
                num_strong_aug.append(else_num)
        else:
            num_weak_aug.extend([int(img_max)] * cls_num)
        return num_weak_aug, num_strong_aug

    def gen_imbalanced_data(self, num_weak_aug, num_strong_aug):
        new_data = []
        new_targets = []
        new_augtype = []
        idx_list = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        self.samples = np.array(self.samples)

        for the_class, the_img_num in zip(classes, num_weak_aug):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            idx_list.append(selec_idx)
            new_data.extend(self.samples[selec_idx])
            new_targets.extend([the_class, ] * the_img_num)
            new_augtype.extend([0, ] * the_img_num)

        for the_class, the_img_num in zip(classes, num_strong_aug):
            idx = idx_list[the_class]
            new_targets.extend([the_class, ] * the_img_num)
            new_augtype.extend([1, ] * the_img_num)
            while the_img_num > 0:
                selec_idx = idx[:the_img_num]
                new_data.extend(self.samples[selec_idx])
                the_img_num -= len(selec_idx)

        new_data = np.array(new_data)
        self.samples = new_data
        self.targets = new_targets
        self.labels = new_targets
        self.augment_type = new_augtype


    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __len__(self):
        return len(self.samples)

    def set_step(self, cur_epoch, total_epoch):
        step = cur_epoch / total_epoch
        if self.step_type == 'step':
            step = step * 3
            self.step = (int)(self.magnitude * step)
        elif self.step_type == 'linear':
            step = 0.2 + 0.8 * step
            self.step = (int)(self.magnitude * step)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_dir, 'images', self.samples[index]))

        target = self.targets[index]
        target = torch.tensor(target).long()
        aug_type = self.augment_type[index]
        if self.transform is not None:
            if aug_type == 1:
                classwise = 1
                if self.step:
                    step = self.step
                else:
                    step = self.magnitude
                if self.classwise_aug:
                    classwise = 1 - (self.alpha * self.factor[target])
                n = (int)(step * classwise)
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
                    ], p=1.0),
                    rand_augment_transform('rand-n{}-m{}-mstd{}'.format(n, 2, 0.5), self.ra_params),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4707, 0.4601, 0.4550], std=[0.2667, 0.2658, 0.2706]),
                ])
            else:
                transform = self.transform[aug_type]
            img = transform(img)
        return img, target


if __name__ == '__main__':
    from torchvision.transforms.functional import InterpolationMode
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in [0.4859, 0.4996, 0.4318]]), )
    strong_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(10, 2), ra_params),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4859, 0.4996, 0.4318], std=[0.1822, 0.1812, 0.1932]),
    ])
    weak_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4859, 0.4996, 0.4318], std=[0.1822, 0.1812, 0.1932])
    ])
    train_transform = [weak_aug, strong_aug]
    train_dataset = DTD(root='/data', train=True, transform=train_transform, imb_factor=0.1)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)

    mean = 0.
    std = 0.
    classes_freq = np.zeros(47)
    for images, y in tqdm.tqdm(train_loader):
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        classes_freq[np.array(y)] += 1
    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)
    print(classes_freq)
    print(mean, std)
