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
from torchvision.transforms.functional import InterpolationMode


class DTD_multi(Dataset):
    def __init__(self, root, train=True, download=False, transform=None, rand_number=0, imb_factor=0.1, imb_type='exp',
                 random_seed=False):
        np.random.seed(rand_number)
        self.random_seed = random_seed
        self.root_dir = os.path.join(root, 'dtd')

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
        self.augment_type = []
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
            num_224, num_else_resol, num_strong_aug = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(num_224, num_else_resol, num_strong_aug)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.samples) / cls_num
        num_224 = []
        num_else_resol = []
        num_strong_aug = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = (int)(img_max * (imb_factor ** (cls_idx / (cls_num - 1.0))))
                if num * 3 > img_max:
                    else_num = (int)(img_max - num)
                else:
                    else_num = (int)(2 * num)
                strong_aug = (int)(img_max - num - else_num)

                num_224.append(num)
                num_else_resol.append(else_num)
                num_strong_aug.append(strong_aug)
                # print(num, else_num, strong_aug, num + else_num + strong_aug)
        else:
            num_224.extend([int(img_max)] * cls_num)
        return num_224, num_else_resol, num_strong_aug

    def gen_imbalanced_data(self, num_224, num_else_resol, num_strong_aug):
        new_data = []
        new_targets = []
        new_augtype = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        self.samples = np.array(self.samples)

        for the_class, the_img_num in zip(classes, num_224):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            self.num_per_cls_dict[the_class] = len(selec_idx)
            new_data.extend(self.samples[selec_idx])
            new_targets.extend([the_class, ] * the_img_num)
            new_augtype.extend([1, ] * the_img_num)

        for the_class, the_img_num in zip(classes, num_else_resol):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            self.num_per_cls_dict[the_class] = len(selec_idx)
            new_data.extend(self.samples[selec_idx])
            new_targets.extend([the_class, ] * the_img_num)
            aug_type = 2 * (np.arange(the_img_num) % 2)
            new_augtype.extend(aug_type)

        for the_class, the_img_num in zip(classes, num_strong_aug):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            self.num_per_cls_dict[the_class] = len(selec_idx)
            new_data.extend(self.samples[selec_idx])
            new_targets.extend([the_class, ] * the_img_num)
            new_augtype.extend([3, ] * the_img_num)

        new_data = np.array(new_data)
        self.samples = new_data
        self.targets = new_targets
        self.augment_type = new_augtype
        self.labels = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_dir, 'images', self.samples[index]))

        target = self.targets[index]
        aug_type = self.augment_type[index]
        target = torch.tensor(target).long()
        if self.transform is not None:
            transform = self.transform[aug_type]
        img = transform(img)
        return img, target#, aug_type



if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_224 = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5273, 0.4702, 0.4235], std=[0.1804, 0.1814, 0.1779])
    ])
    transform_128 = transforms.Compose([
        transforms.RandomResizedCrop(128, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5273, 0.4702, 0.4235], std=[0.1804, 0.1814, 0.1779])
    ])
    transform_480 = transforms.Compose([
        transforms.RandomResizedCrop(480, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5273, 0.4702, 0.4235], std=[0.1804, 0.1814, 0.1779])
    ])
    transform_strong = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5273, 0.4702, 0.4235], std=[0.1804, 0.1814, 0.1779])
    ])

    train_transform = [transform_128, transform_224, transform_480, transform_strong]

    train_dataset = DTD_multi(root='/home/vision/jihun/work/paco/data', train=True, download=False, transform=train_transform,
                        imb_factor=0.1)
    # test_dataset = DTD_multi(root='./data', train=False, download=False, transform=train_transform)

    # print(train_dataset.get_cls_num_list())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=1, shuffle=False,
    #     num_workers=0, persistent_workers=False, pin_memory=True)

    # for i in range(len(train_dataset.get_cls_num_list())):
    #     images = torch.empty(train_dataset.get_cls_num_list()[0], 3, 224, 224)
    #     idx = 0
    #     for image, y in train_loader:
    #         if y == i:
    #             images[idx] = image
    #             idx += 1
    #
    #     plt.figure()
    #     plt.title(f'{i}')
    #     plt.clf()
    #     plt.imshow(torchvision.utils.make_grid(images, normalize=True).permute(1, 2, 0))
    #     plt.savefig(f'DTD_{i}.png')

    # classes_freq = np.zeros(47)
    # for x, y in tqdm.tqdm(test_loader):
    #     classes_freq[np.array(y)] += 1
    # print(classes_freq)

    mean = 0.
    std = 0.
    classes_freq = np.zeros(47)
    for images, y in tqdm.tqdm(train_loader):
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        classes_freq[np.array(y)] += 1
        # print(aug_type)
    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)
    print(classes_freq)
    print(mean, std)
