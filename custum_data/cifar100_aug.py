import random

import torch
import torchvision
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
from PIL import Image
from dataset.randaugment import rand_augment_transform


class ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True, random_seed=False, classwise_aug=False,
                 alpha=0.5, magnitude=10, step_type=False):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.random_seed = random_seed
        np.random.seed(rand_number)
        self.classwise_aug = classwise_aug
        self.magnitude = magnitude

        self.alpha = alpha
        self.step_type = step_type
        self.step = None
        num_weak_aug, num_strong_aug = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(num_weak_aug, num_strong_aug)
        self.ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x))
                                                                               for x in [0.4914, 0.4822, 0.4465]]), )
        self.cls_num_list = self.get_cls_num_list()
        self.factor = np.array(self.cls_num_list) / self.cls_num_list[0]

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        num_weak_aug = []
        num_strong_aug = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = (int)(img_max * (imb_factor ** (cls_idx / (cls_num - 1.0))))
                num_weak_aug.append(num)
                else_num = (int)(img_max) - num
                num_strong_aug.append(else_num)
                # print(num,else_num, img_max)
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
        for the_class, the_img_num in zip(classes, num_weak_aug):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            idx_list.append(selec_idx)
            new_data.append(self.data[selec_idx])
            new_targets.extend([the_class, ] * the_img_num)
            new_augtype.extend([0, ] * the_img_num)

        for the_class, the_img_num in zip(classes, num_strong_aug):
            new_targets.extend([the_class, ] * the_img_num)
            new_augtype.extend([1, ] * the_img_num)
            idx = idx_list[the_class]
            while the_img_num > 0:
                np.random.shuffle(idx)
                selec_idx = idx[:the_img_num]
                new_data.append(self.data[selec_idx])
                the_img_num -= len(selec_idx)

        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        self.augment_type = new_augtype

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def set_step(self, cur_epoch, total_epoch):
        step = cur_epoch / total_epoch
        if self.step_type == 'step':
            step = step * 3
            self.step = (int)(self.magnitude * step)
        elif self.step_type == 'linear':
            step = 0.2 + 0.8 * step
            self.step = (int)(self.magnitude * step)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        aug_type = self.augment_type[index]
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
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        sample = transform(img)

        return sample, target


class ImbalanceCIFAR100(ImbalanceCIFAR10):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100
