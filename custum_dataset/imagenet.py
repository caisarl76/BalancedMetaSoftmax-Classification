import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageNetLT(Dataset):

    def __init__(self, root, txt=None, train=None, transform=None, new_class_idx_sorted=None, random_seed=False):
        self.random_seed = random_seed
        self.img_path = []
        self.targets = []
        self.transform = transform
        root = os.path.join(root, 'imagenet')
        if not txt:
            if train:
                txt = os.path.join(root, 'ImageNet_LT_train.txt')
            else:
                txt = os.path.join(root, 'ImageNet_LT_test.txt')
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.img_path = np.array(self.img_path)
        self.targets = np.array(self.targets)
        num_in_class = []
        for class_idx in np.unique(self.targets):
            num_in_class.append(len(np.where(self.targets == class_idx)[0]))
        self.num_in_class = num_in_class

        # self.sort_dataset(new_class_idx_sorted)

        self.cls_num_list = [np.sum(np.array(self.targets) == i) for i in range(1000)]

    def get_cls_num_list(self):
        return self.cls_num_list

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.targets[index]

        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            if isinstance(self.transform, list):
                if type(self.transform) == list:
                    samples = []
                    for transform in self.transform:
                        sample = transform(img)
                        samples.append(sample)
                    return samples, label
            else:
                sample = self.transform(img)

        return sample, label  # , index

    def sort_dataset(self, new_class_idx_sorted=None):
        idx = np.argsort(self.targets)
        self.targets = self.targets[idx]
        self.img_path = self.img_path[idx]
        if new_class_idx_sorted is None:
            new_class_idx_sorted = np.argsort(self.num_in_class)[::-1]
        for idx, target in enumerate(self.targets):
            self.targets[idx] = np.where(new_class_idx_sorted == target)[0]
        idx = np.argsort(self.targets)
        self.targets = self.targets[idx]
        self.img_path = self.img_path[idx]
        self.new_class_idx_sorted = new_class_idx_sorted

if __name__ == '__main__':
    train_dataset = ImageNetLT('/home/vision/jihun/work/binlt/data/',train=True)
    print(len(train_dataset))
    print(train_dataset.get_cls_num_list()[-1])
    tmp_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=None,
        num_workers=0, pin_memory=True, sampler=None)
    cls_num_list = train_dataset.get_cls_num_list()
    print('total classes:', len(cls_num_list))
    print('max classes:', max(cls_num_list))
    print('train loader length:', len(tmp_loader))
    train_loader_length = len(tmp_loader)
    oversampled_loader_length = len(cls_num_list) * max(cls_num_list)
    dataset_ratio = train_loader_length / oversampled_loader_length
    print('dataset ratio:', dataset_ratio)
    print(int(dataset_ratio * 100))
    del train_dataset