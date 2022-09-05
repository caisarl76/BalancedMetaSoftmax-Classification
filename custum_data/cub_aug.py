import os
import random

import numpy as np
import pandas as pd
import torch.utils.data
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from dataset.randaugment import rand_augment_transform


class Cub2011(Dataset):
    base_folder = 'images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, imb_type='exp', imb_factor=0.1, transform=None, rand_number=0,
                 random_seed=False, classwise_aug=False, alpha=0.5, magnitude=10, step_type=False):
        self.random_seed = random_seed
        np.random.seed(rand_number)
        self.classwise_aug = classwise_aug
        self.magnitude = magnitude

        self.alpha = alpha
        self.step_type = step_type
        self.step = None
        self.root = os.path.expanduser(root)
        self.root = os.path.join(os.getcwd(), self.root, 'cub')
        self.transform = transform
        self.loader = default_loader
        self.train = train
        num_weak_aug, num_strong_aug = self.get_img_num_per_cls(200, imb_type, imb_factor)
        self.img_num_list = num_weak_aug
        self.gen_imbalanced_data(num_weak_aug, num_strong_aug)
        self.ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x))
                                                                               for x in [0.4859, 0.4996, 0.4318]]), )

        self.cls_num_list = self.get_cls_num_list()
        self.factor = np.array(self.cls_num_list) / self.cls_num_list[0]

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = 30
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
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(
            os.path.join(self.root, 'image_class_labels.txt'),
            sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')
        self.aug_type = []
        idx_list = []
        self.data = []
        self.targets = []
        if self.train:
            train = data[data.is_training_img == 1]
            classes = np.unique(train.target)
            for the_class, the_img_num in zip(classes, num_weak_aug):
                if len(train[train.target==the_class]) != 30:
                    idx = np.random.choice(29, the_img_num)
                else:
                    idx = np.random.choice(30, the_img_num)

                # np.random.shuffle(idx)
                # selec_idx = idx[:the_img_num]
                idx_list.append(idx)
                data_path = train[train.target == the_class].iloc[idx]['filepath'].to_numpy()
                self.data.extend(data_path)
                self.targets.extend([the_class, ] * the_img_num)
                self.aug_type.extend([0, ] * the_img_num)

            for the_class, the_img_num in zip(classes, num_strong_aug):
                self.targets.extend([the_class, ] * the_img_num)
                self.aug_type.extend([0, ] * the_img_num)
                idx = idx_list[the_class-1]
                while the_img_num > 0:
                    np.random.shuffle(idx)
                    selec_idx = idx[:the_img_num]
                    data_path = train[train.target == the_class].iloc[selec_idx]['filepath'].to_numpy()
                    self.data.extend(data_path)
                    the_img_num -= len(data_path)
        else:
            self.data = data[data.is_training_img == 0]

    def get_cls_num_list(self):
        return self.img_num_list

    def __len__(self):
        return len(self.data)

    def set_step(self, cur_epoch, total_epoch):
        step = cur_epoch / total_epoch
        if self.step_type == 'step':
            step = step * 3
            self.step = (int)(self.magnitude * step)
        elif self.step_type == 'linear':
            step = 0.2 + 0.8 * step
            self.step = (int)(self.magnitude * step)

    def __getitem__(self, idx):
        path = self.data[idx]
        path = os.path.join(self.root, self.base_folder, path)
        target = self.targets[idx] -1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        aug_type = self.aug_type[idx]
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
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = Cub2011(root='/media/hd/jihun/data/CUB_200_2011', train=True, transform=train_transform)
    test_dataset = Cub2011(root='/media/hd/jihun/data/CUB_200_2011', train=False, transform=train_transform)
    print(len(train_dataset))
    print(len(test_dataset))
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)
    for i, (image, target) in enumerate(val_loader):
        print(image.shape, target.shape)
        break
