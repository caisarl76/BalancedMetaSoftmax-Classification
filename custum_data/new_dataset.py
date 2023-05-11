import os
import argparse

import json
import torch.optim
from torchvision import datasets, transforms

if __name__ == '__main__':
    from fruits import Fruits
    from places import PlacesLT
    from dtd import DTD
    from aircraft import Aircraft
    from cars import Cars
    from dogs import Dogs
    from flowers import Flowers
    from imabalance_cub import Cub2011
    from imbalance_cifar import ImbalanceCIFAR100, ImbalanceCIFAR10
    from inat import INaturalist
    from imagenet import ImageNetLT
    from caltech101 import ImbalanceCaltech101
else:
    from .fruits import Fruits
    from .places import PlacesLT
    from .dtd import DTD
    from .aircraft import Aircraft
    from .cars import Cars
    from .dogs import Dogs
    from .flowers import Flowers
    from .imabalance_cub import Cub2011
    from .imbalance_cifar import ImbalanceCIFAR100, ImbalanceCIFAR10
    from .inat import INaturalist
    from .imagenet import ImageNetLT
    from .caltech101 import ImbalanceCaltech101

class_dict = {
    'cub': [200, [0, 72], [72, 142], [142, 200], [0.4859, 0.4996, 0.4318], [0.1822, 0.1812, 0.1932], Cub2011],
    'cifar10': [10, [0, 3], [3, 7], [7, 10], [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010], ImbalanceCIFAR10],
    'cifar100': [100, [0, 36], [36, 71], [71, 100], [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010],
                 ImbalanceCIFAR100],
    'fgvc': [100, [0, 36], [36, 71], [71, 100], [0.4796, 0.5107, 0.5341], [0.1957, 0.1945, 0.2162], Aircraft],
    'dogs': [120, [0, 43], [43, 85], [85, 120], [0.4765, 0.4517, 0.3911], [0.2342, 0.2293, 0.2274], Dogs],
    'cars': [196, [0, 70], [70, 139], [139, 196], [0.4707, 0.4601, 0.4550], [0.2667, 0.2658, 0.2706], Cars],
    'flowers': [102, [0, 36], [36, 72], [72, 102], [0.4344, 0.3830, 0.2954], [0.2617, 0.2130, 0.2236], Flowers],
    'dtd': [47, [0, 14], [14, 33], [33, 47], [0.5273, 0.4702, 0.4235], [0.1804, 0.1814, 0.1779], DTD],
    'caltech101': [102, [0, 36], [36, 72], [72, 102], [0.5494, 0.5232, 0.4932], [0.2463, 0.2461, 0.2460],
                   ImbalanceCaltech101],
    'fruits': [24, [0, 7], [7, 14], [14, 24], [0.6199, 0.5188, 0.4020], [0.2588, 0.2971, 0.3369], Fruits],
    'places': [365, [0, 131], [131, 259], [259, 365], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], PlacesLT],
    'imagenet': [1000, [0, 390], [390, 835], [835, 1000], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], ImageNetLT],
    'inat': [8142, [0, 842], [842, 4543], [4543, 8142], [0.466, 0.471, 0.380], [0.195, 0.194, 0.192], INaturalist],
    'iNaturalist18': [8142, [0, 842], [842, 4543], [4543, 8142], [0.466, 0.471, 0.380], [0.195, 0.194, 0.192], INaturalist],
}
smooth_dict = {
    'caltech101_10': [0.01, 0.01],
    'cifar10_10': [0.07, 0.07],
    'cifar10_100': [0.05, 0.1],
    'cifar100_10': [0.08, 0.36],
    'cifar100_100': [0.08, 0.36],
    'cub_10': [0.04, 0.01],
    'dtd_10': [0.01, 0.39],
    'dogs_10': [0.0, 0.03],
    'cars_10': [0.0, 0.38],
    'flowers_10': [0.04, 0.04],
    'fgvc_10': [0.0, 0.39],
    'fruits_100': [0.0, 0.0],
    'places': [0.4, 0.1],
    'imagenet': [0.3, 0.0],
    'inat': [0.3, 0.0],
}


def dataset_info(args):
    args.num_classes, args.head_class_idx, args.med_class_idx, args.tail_class_idx, MEAN, STD, data_class \
        = class_dict[args.dataset]
    return args, MEAN, STD, data_class


def get_dataset(phase, data_root='./dataset', dataset='cub', sampler_dic=None, batch_size=128, num_workers=2,
                imb_ratio=0.1, random_seed=None, meta=False):
    dataset = dataset.lower()
    if phase not in ['meta', 'train', 'train_plain']:
        imb_ratio = 1.0
    if '_' in dataset:
        dataset = dataset.split('_')[0]
    num_classes, _, _, _, MEAN, STD, data_class = class_dict[dataset]

    if 'cifar10' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

    # print(transform_train)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    # root = os.path.join(data_root, dataset)
    # print(val_transform)

    if dataset in ['places', 'imagenet', 'inat', 'iNaturalist18']:
        train_dataset = data_class(root=data_root,
                                   train=True,
                                   transform=transform_train,
                                   random_seed=random_seed,
                                   meta=meta)
    else:
        train_dataset = data_class(root=data_root,
                                   imb_factor=imb_ratio,
                                   train=True,
                                   transform=transform_train,
                                   )
    if not os.path.exists('cls_freq'):
        os.makedirs('cls_freq')
    freq_path = os.path.join('cls_freq', dataset + '.json')
    cls_num_list = train_dataset.get_cls_num_list()
    with open(freq_path, 'w') as fd:
        json.dump(cls_num_list, fd)

    if dataset in ['cars', 'dogs', 'fruits']:
        new_class_idx = train_dataset.get_new_class_idx_sorted()
        val_dataset = data_class(root=data_root,
                                 train=False,
                                 transform=val_transform,
                                 new_class_idx_sorted=new_class_idx)
    else:
        val_dataset = data_class(root=data_root,
                                 train=False,
                                 transform=val_transform)


    if sampler_dic and phase == 'train' and sampler_dic.get('batch_sampler', False):
        print('Using sampler: ', sampler_dic['sampler'])
        data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                  batch_sampler=sampler_dic['sampler'](train_dataset,
                                                                                       **sampler_dic['params']),
                                                  num_workers=num_workers)
    elif phase == 'train' and not meta:
        data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers)

    elif sampler_dic and (phase == 'train' or meta):
        print('Using sampler: ', sampler_dic['sampler'])
        print('Sampler parameters: ', sampler_dic['params'])
        data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                                  sampler=sampler_dic['sampler'](train_dataset, **sampler_dic['params']),
                                                  num_workers=num_workers)
    else:
        if phase in ['val', 'test']:
            dataset = val_dataset
        else:
            dataset = train_dataset
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False if phase in ['val', 'test'] else True,
            num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='flowers',
                        choices=['inat', 'places', 'cub', 'cifar10', 'cifar100', 'fgvc', 'dogs', 'cars', 'flowers',
                                 'caltech101'])
    parser.add_argument('--imb_ratio', default=0.1, type=float)
    parser.add_argument('--data', metavar='DIR', default='../data/')
    args = parser.parse_args()

    import numpy as np

    for data in ['inat', 'places', 'imagenet']:
        for resol in [(480, 224, 128)]:
            args.dataset = data
            # args.imb_ratio = ratio
            print(args.dataset, args.imb_ratio, end=' ')
            train_dataset, val_dataset = get_dataset(args, train_img_size=resol, random_seed=True)
            print(len(train_dataset), len(val_dataset))
            print(train_dataset.get_cls_num_list()[-1])
            tmp_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=1, shuffle=None,
                num_workers=args.workers, pin_memory=True, sampler=None)
            cls_num_list = train_dataset.get_cls_num_list()
            print('total classes:', len(cls_num_list))
            print('max classes:', max(cls_num_list))
            print('train loader length:', len(tmp_loader))
            train_loader_length = len(tmp_loader)
            oversampled_loader_length = len(cls_num_list) * max(cls_num_list)
            dataset_ratio = train_loader_length / oversampled_loader_length
            print('dataset ratio:', dataset_ratio)
            print(int(dataset_ratio * 100))
            del train_dataset, val_dataset
    from utils.mislas import ClassAwareSampler
    from collections import Counter

    # train_dataset, val_dataset = get_dataset(args, train_img_size=(480, 224, 128), random_seed=True)
    #
    # cls_num_list = np.asarray(train_dataset.get_cls_num_list())
    # cls_num_list = cls_num_list / cls_num_list[0]
    #
    # balance_sampler = ClassAwareSampler(train_dataset)
    #
    # loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=512, shuffle=False,
    #     num_workers=0, sampler=balance_sampler)
    # img, label = next(iter(loader))
    # cnt = Counter(label.tolist())
    # idx = 0
    # resol_idx = []
    # while idx < len(label):
    #     class_idx = label[idx].item()
    #     num_samples = cnt[class_idx]
    #     imb_ratio = cls_num_list[class_idx]
    #     if imb_ratio < 0.34:
    #         idx_else = 2 * (int)(np.ceil(num_samples * cls_num_list[class_idx]))
    #         idx_224 = num_samples - idx_else
    #     else:
    #         idx_224 = (int)(num_samples * cls_num_list[class_idx])
    #         idx_else = num_samples - idx_224
    #     resol_idx.extend([1] * idx_224)
    #     temp = 2 * (np.arange(idx_else) % 2)
    #     resol_idx.extend(temp)
    #     idx += num_samples
    # image_128 = []
    # image_224 = []
    # image_480 = []
    # for i in range(len(img[0])):
    #     image = img[resol_idx[i]][i]
    #     if resol_idx[i] == 0:
    #         image_128.append(image)
    #     elif resol_idx[i] == 1:
    #         image_224.append(image)
    #     elif resol_idx[i] == 2:
    #         image_480.append(image)
    #     else:
    #         print('wrong idx ', resol_idx[i])
    #         break
    # image_128 = torch.stack(image_128, 0)
    # image_224 = torch.stack(image_224, 0)
    # image_480 = torch.stack(image_480, 0)
    # print(image_128.shape, image_224.shape, image_480.shape)
    #
    # for data in ['inat', 'places', 'imagenet']:
    # # for data in ['caltech101', 'cub', 'dogs', 'dtd', 'fgvc', 'flowers']:
    #     for resol in [(480, 224, 128)]:
    #         args.dataset = data
    #         print(args.dataset, end=' ')
    #         train_dataset, val_dataset = get_strongaug_dataset(args, train_img_size=resol, val_img_size=resol)
    #         print(len(train_dataset), len(val_dataset))
    #         print(train_dataset.get_cls_num_list()[-1])
    #
    #         del train_dataset, val_dataset
    # for data in ['cifar10', 'cifar100']:
    #     for ratio in [0.1, 0.01]:
    #         for resol in [(480, 224, 128)]:
    #             args.dataset = data
    #             args.imb_ratio = ratio
    #             print(args.dataset, args.imb_ratio, end=' ')
    #             train_dataset, val_dataset = get_dataset(args, train_img_size=resol, random_seed=True)
    #             print(len(train_dataset), len(val_dataset))
    #             print(train_dataset.get_cls_num_list()[-1])
    #             del train_dataset, val_dataset
