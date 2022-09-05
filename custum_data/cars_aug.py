import os
import random

import numpy as np
import scipy.io as sio
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
import tqdm

from dataset.randaugment import rand_augment_transform


class Cars(VisionDataset):
    file_list = {
        'train_imgs': ('http://ai.stanford.edu/~jkrause/car196/cars_train.tgz', 'cars_train'),
        'train_annos': (
            'http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz', 'car_devkit/devkit/cars_train_annos.mat'),
        'test_imgs': ('http://ai.stanford.edu/~jkrause/car196/cars_test.tgz', 'cars_test'),
        'test_annos': (
            'http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz', 'car_devkit/devkit/cars_test_annos_withlabels.mat'),
        'meta': ('', 'car_devkit/devkit/cars_meta.mat'),
        'annos': ('', 'car_devkit/devkit/cars_annos.mat')
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, rand_number=0,
                 imb_factor=0.1, imb_type='exp', new_class_idx_sorted=None, random_seed=False, classwise_aug=False,
                 alpha=0.5, magnitude=10, step_type=False):
        super(Cars, self).__init__(root, transform=transform, target_transform=target_transform)
        self.random_seed = random_seed
        self.classwise_aug = classwise_aug
        self.magnitude = magnitude
        self.alpha = alpha
        self.step_type = step_type
        self.step = None

        self.transform = transform
        np.random.seed(rand_number)
        self.loader = default_loader
        self.train = train
        self.root = os.path.join(os.getcwd(), self.root, 'cars')

        class_names = np.array(sio.loadmat(os.path.join(self.root, self.file_list['meta'][1]))['class_names'])

        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        if self.train:
            loaded_mat = sio.loadmat(os.path.join(self.root, self.file_list['train_annos'][1]))
            loaded_mat = loaded_mat['annotations'][0]
            self.root += '/cars_train'
        else:
            loaded_mat = sio.loadmat(os.path.join(self.root, self.file_list['test_annos'][1]))
            loaded_mat = loaded_mat['annotations'][0]
            self.root += '/cars_test'
        self.samples = []
        for item in loaded_mat:
            path = str(item[-1][0])
            label = int(item[-2][0]) - 1
            self.samples.append((path, label))
        self.samples = np.array(self.samples)
        self.targets = np.array(self.samples[:, 1])
        self.targets = self.targets.astype(np.int)

        num_in_class = []
        for class_idx in np.unique(self.targets):
            num_in_class.append(len(np.where(self.targets == class_idx)[0]))
        self.num_in_class = num_in_class

        self.sort_dataset(new_class_idx_sorted)
        self.class_names = class_names[0][self.new_class_idx_sorted]
        # print(self.class_names)
        self.classes = np.unique(self.targets)
        self.cls_num = len(self.classes)
        if train:
            num_weak_aug, num_strong_aug = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(num_weak_aug, num_strong_aug)
        self.ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x))
                                                                               for x in [0.4707, 0.4601, 0.4550]]), )
        self.cls_num_list = self.get_cls_num_list()
        self.factor = np.array(self.cls_num_list) / self.cls_num_list[0]

    def sort_dataset(self, new_class_idx_sorted=None):
        idx = np.argsort(self.targets)
        self.targets = self.targets[idx]
        self.samples = self.samples[idx]
        if new_class_idx_sorted is None:
            new_class_idx_sorted = np.argsort(self.num_in_class)[::-1]
        for idx, target in enumerate(self.targets):
            self.targets[idx] = np.where(new_class_idx_sorted == target)[0]
        idx = np.argsort(self.targets)
        self.targets = self.targets[idx]
        self.samples = self.samples[idx]
        for idx, sample in enumerate(self.samples):
            self.samples[idx][1] = self.targets[idx]
        self.new_class_idx_sorted = new_class_idx_sorted
        # tmp = np.zeros(196)
        # for sample in self.samples:
        #     tmp[int(sample[1])] += 1
        # print(tmp)

    def get_new_class_idx_sorted(self):
        return self.new_class_idx_sorted

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # img_max = len(self.samples) / cls_num
        # manually select max frequency to be that of second class
        img_max = max(sorted(self.num_in_class)[::-1][1:])
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
            new_data.append(self.samples[selec_idx])
            new_targets.extend([the_class, ] * the_img_num)
            new_augtype.extend([0, ] * the_img_num)
        for the_class, the_img_num in zip(classes, num_strong_aug):
            new_targets.extend([the_class, ] * the_img_num)
            new_augtype.extend([1, ] * the_img_num)
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data[the_class] = np.append(new_data[the_class], self.samples[selec_idx], axis=0)
            resample = the_img_num - len(selec_idx)
            while resample > 0:
                np.random.shuffle(idx)
                selec_idx = idx[:resample]
                new_data[the_class] = np.append(new_data[the_class], self.samples[selec_idx], axis=0)
                resample -= len(selec_idx)

        new_data = np.vstack(new_data)
        self.samples = new_data
        self.targets = new_targets
        self.labels = new_targets
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
        path, target = self.samples[index]
        path = os.path.join(self.root, path)
        target = torch.tensor(int(target)).long()
        aug_type = self.augment_type[index]
        image = self.loader(path)
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
        image = transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.file_list['train_imgs'][1]))
                and os.path.exists(os.path.join(self.root, self.file_list['train_annos'][1]))
                and os.path.exists(os.path.join(self.root, self.file_list['test_annos'][1]))
                and os.path.exists(os.path.join(self.root, self.file_list['test_imgs'][1])))

    def _download(self):
        print('Downloading...')
        for url, filename in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
        print('Extracting...')
        archive = os.path.join(self.root, self.file_list['imgs'][1])
        extract_archive(archive)


if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=1, shuffle=False,
    #     num_workers=0, persistent_workers=False, pin_memory=True)
    # for i in range(len(train_dataset.get_cls_num_list())):
    #     images = torch.empty(train_dataset.get_cls_num_list()[0], 3, 224, 224)
    #     idx = 0
    #     for image, y in train_loader:
    #         if y == i:
    #             images[idx] = image
    #             idx += 1
    #
    #     plt.figure(dpi=400)
    #     plt.title(f'{i}')
    #     plt.clf()
    #     plt.imshow(torchvision.utils.make_grid(images, normalize=True).permute(1, 2, 0))
    #     plt.savefig(f'Cars_train{i}.png')

    train_dataset = Cars('/data', train=True, download=True, transform=train_transform, imb_factor=0.1)
    new_class_idx = train_dataset.get_new_class_idx_sorted()
    test_dataset = Cars('/data', train=False, download=True, new_class_idx_sorted=new_class_idx,
                        transform=train_transform, imb_factor=1)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)

    # classes_freq = np.zeros(train_dataset.cls_num)
    # for x, y in tqdm.tqdm(train_loader):
    #     classes_freq[np.array(y)] += 1
    # print(classes_freq)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=True)

    # classes_freq = np.zeros(train_dataset.cls_num)
    # for x, y in tqdm.tqdm(test_loader):
    #     classes_freq[np.array(y)] += 1
    # print(classes_freq)
    # images = torch.empty(train_dataset.get_cls_num_list()[0], 3, 224, 224)
    # for i in range(len(train_dataset.get_cls_num_list())):
    #     images = torch.empty(train_dataset.get_cls_num_list()[0], 3, 224, 224)
    #     idx = 0
    #     for image, y in test_loader:
    #         if y == i:
    #             images[idx] = image
    #             idx += 1
    #
    #     plt.figure(dpi=400)
    #     plt.title(f'{i}')
    #     plt.clf()
    #     plt.imshow(torchvision.utils.make_grid(images, normalize=True).permute(1, 2, 0))
    #     plt.savefig(f'Cars_val{i}.png')

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=128, shuffle=False,
    #     num_workers=0, persistent_workers=False, pin_memory=True)
    # for images, y in train_loader:
    #     print(y)

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=1, shuffle=False,
    #     num_workers=0, persistent_workers=False, pin_memory=True)
    #
    # print(train_dataset.get_cls_num_list())
    # print(sum(train_dataset.get_cls_num_list()))
    mean = 0.
    std = 0.
    classes_freq = np.zeros(196)
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
