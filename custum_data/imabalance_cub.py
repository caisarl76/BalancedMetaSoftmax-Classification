import os
import numpy as np
import pandas as pd
import torch.utils.data
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

transform_train_cub = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform_cub = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Cub2011(Dataset):
    base_folder = 'images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, imb_type='exp', imb_factor=0.1, transform=None, rand_number=0):
        np.random.seed(rand_number)

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.img_num_list = self.get_img_num_per_cls(200, imb_type, imb_factor)
        self.gen_imbalanced_data()
        self.cls_num_list = self.img_num_list
        print('none')

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = 30
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(
            os.path.join(self.root, 'image_class_labels.txt'),
            sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        for i, item in enumerate(image_class_labels.target):
            image_class_labels.target[i] = item - 1
        data = images.merge(image_class_labels, on='img_id')

        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            train = self.data[self.data.is_training_img == 1]
            self.data = train[train.target == 0].iloc[np.random.choice(30, self.img_num_list[0]), :]
            for i in range(1, 200):
                temp = train[train.target == i].iloc[
                       np.random.choice(len(train[train.target == i]), self.img_num_list[i]), :]
                self.data = self.data.append(temp)
        else:
            self.data = self.data[self.data.is_training_img == 0]
        self.targets = self.data.target.to_numpy()

    def get_cls_num_list(self):
        return self.img_num_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        # target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            if type(self.transform) == list:
                sample1 = self.transform[0](img)
                sample2 = self.transform[0](img)
                return [sample1, sample2], sample.target
            else:
                img = self.transform(img)
                return img, sample.target, idx


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
