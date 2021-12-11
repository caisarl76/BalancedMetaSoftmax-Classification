import argparse
import os

import PIL
import torch.optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

CUB_MEAN = [0.4859, 0.4996, 0.4318]
CUB_STD = [0.1822, 0.1812, 0.1932]

# CUB_MEAN = [0.485, 0.456, 0.406]
# CUB_STD = [0.229, 0.224, 0.225]

INAT_MEAN = [0.466, 0.471, 0.380]
INAT_STD = [0.195, 0.194, 0.192]

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

FGVC_MEAN = [0.4796, 0.5107, 0.5341]
FGVC_STD = [0.1957, 0.1945, 0.2162]

DOGS_MEAN = [0.4765, 0.4517, 0.3911]
DOGS_STD = [0.2342, 0.2293, 0.2274]

CARS_MEAN = [0.4707, 0.4601, 0.4550]
CARS_STD = [0.2667, 0.2658, 0.2706]

FLOWERS_MEAN = [0.4344, 0.3830, 0.2954]
FLOWERS_STD = [0.2617, 0.2130, 0.2236]

DTD_MEAN = [0.5273, 0.4702, 0.4235]
DTD_STD = [0.1804, 0.1814, 0.1779]

PLACES_MEAN = [0.485, 0.456, 0.406]
PLACES_STD = [0.229, 0.224, 0.225]

CALTECH101_MEAN = [0.5494, 0.5232, 0.4932]
CALTECH101_STD = [0.2463, 0.2461, 0.2460]

FRUIT360_MEAN = [0.6199, 0.5188, 0.4020]
FRUIT360_STD = [0.2588, 0.2971, 0.3369]

def get_dataset(data_root, dataset,
                sampler_dic=None, meta=False,
                batch_size=128, num_workers=4,
                transform_train=None, val_transform=None, imb_ratio=0.1):
    if dataset == 'places':
        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=PLACES_MEAN, std=PLACES_STD),
            ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=PLACES_MEAN, std=PLACES_STD))

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=PLACES_MEAN, std=PLACES_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=PLACES_MEAN, std=PLACES_STD))

        train_dataset = PlacesLT(
            root=data_root,
            train=True,
            transform=transform_train)
        val_dataset = PlacesLT(
            root=data_root,
            train=False,
            transform=val_transform)

    elif dataset == 'imagenet':
        data_root = os.path.join(data_root, 'imagenet')
        txt_train = f'./data/imagenet/ImageNet_LT_train.txt'
        txt_test = f'./data/imagenet/ImageNet_LT_test.txt'
        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD),
            ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD))
        print(transform_train)
        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD),
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD))
        print(val_transform)
        train_dataset = ImageNetLT(
            root=data_root,
            txt=txt_train,
            transform=transform_train
        )
        val_dataset = ImageNetLT(
            root=data_root,
            txt=txt_test,
            transform=val_transform
        )

    elif dataset == 'inat':
        data_root = os.path.join(data_root, 'inat')

        txt_train = os.path.join(data_root, 'iNaturalist18_train.txt')
        txt_test = os.path.join(data_root, 'iNaturalist18_val.txt')
        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=INAT_MEAN, std=INAT_STD)
            ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=INAT_MEAN, std=INAT_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=INAT_MEAN, std=INAT_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=INAT_MEAN, std=INAT_STD))
        print(val_transform)

        train_dataset = INaturalist(
            root=data_root,
            txt=txt_train,
            transform=transform_train
        )
        val_dataset = INaturalist(
            root=data_root,
            txt=txt_test,
            transform=val_transform
        )
    elif dataset == 'cub':
        data_root = os.path.join(data_root, 'cub')
        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=CUB_MEAN, std=CUB_STD)
            ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=CUB_MEAN, std=CUB_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=CUB_MEAN, std=CUB_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=CUB_MEAN, std=CUB_STD))
        print(val_transform)

        train_dataset = Cub2011(
            root=data_root,
            imb_type='exp',
            imb_factor=imb_ratio,
            train=True,
            transform=transform_train
        )
        val_dataset = Cub2011(
            root=data_root,
            train=False,
            transform=val_transform
        )

    elif dataset == 'cifar10':
        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.Resize(size=(224, 224), interpolation=PIL.Image.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
            ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD))
        print(val_transform)

        train_dataset = ImbalanceCIFAR10(root=data_root, imb_type='exp', imb_factor=imb_ratio, rand_number=0,
                                         train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=val_transform
        )

    elif dataset == 'cifar100':

        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.Resize(size=(224, 224), interpolation=PIL.Image.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
            ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD))
        print(val_transform)

        train_dataset = ImbalanceCIFAR100(root=data_root, imb_type='exp', imb_factor=imb_ratio, rand_number=0,
                                          train=True, download=True, transform=transform_train)
        val_dataset = ImbalanceCIFAR100(root=data_root, imb_factor=1.0,
                                          train=False, download=True, transform=val_transform)

    elif dataset == 'fgvc':

        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=FGVC_MEAN, std=FGVC_STD),
            ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=FGVC_MEAN, std=FGVC_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=FGVC_MEAN, std=FGVC_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=FGVC_MEAN, std=FGVC_STD))
        print(val_transform)

        train_dataset = Aircraft(root=data_root, imb_type='exp', imb_factor=imb_ratio, train=True, download=True, transform=transform_train)
        val_dataset = Aircraft(root=data_root, imb_type='exp', imb_factor=imb_ratio, train=False, download=True, transform=val_transform)

    elif dataset == 'dogs':

        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=DOGS_MEAN, std=DOGS_STD),
            ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=DOGS_MEAN, std=DOGS_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=DOGS_MEAN, std=DOGS_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=DOGS_MEAN, std=DOGS_STD))
        print(val_transform)

        train_dataset = Dogs(root=data_root, imb_type='exp', imb_factor=imb_ratio, train=True, download=True, transform=transform_train)
        val_dataset = Dogs(root=data_root, imb_type='exp', imb_factor=imb_ratio, train=False, download=True, transform=val_transform)

    elif dataset == 'cars':

        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=CARS_MEAN, std=CARS_STD),
            ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=CARS_MEAN, std=CARS_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=CARS_MEAN, std=CARS_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=CARS_MEAN, std=CARS_STD))
        print(val_transform)

        train_dataset = Cars(root=data_root, imb_type='exp', imb_factor=imb_ratio, train=True, download=True, transform=transform_train)
        new_class_idx = train_dataset.get_new_class_idx_sorted()
        val_dataset = Cars(root=data_root, imb_type='exp', imb_factor=imb_ratio, train=False, download=True, transform=val_transform, new_class_idx_sorted=new_class_idx)

    elif dataset == 'flowers':

        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=FLOWERS_MEAN, std=FLOWERS_STD),
            ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=FLOWERS_MEAN, std=FLOWERS_STD))
        print(transform_train)

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=FLOWERS_MEAN, std=FLOWERS_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=FLOWERS_MEAN, std=FLOWERS_STD))
        print(val_transform)

        train_dataset = Flowers(root=data_root, imb_type='exp', imb_factor=imb_ratio, train=True, download=True, transform=transform_train)
        val_dataset = Flowers(root=data_root, imb_type='exp', imb_factor=imb_ratio, train=False, download=True, transform=val_transform)

    elif dataset == 'dtd':

        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=DTD_MEAN, std=DTD_STD),
            ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=DTD_MEAN, std=DTD_STD))
        print(transform_train)
        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=DTD_MEAN, std=DTD_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=DTD_MEAN, std=DTD_STD))
        print(val_transform)

        train_dataset = DTD(root=data_root, imb_type='exp', imb_factor=imb_ratio, train=True, download=True, transform=transform_train)
        val_dataset = DTD(root=data_root, imb_type='exp', imb_factor=imb_ratio, train=False, download=True, transform=val_transform)

    elif dataset == 'caltech101':

        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=CALTECH101_MEAN, std=CALTECH101_STD),
            ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=CALTECH101_MEAN, std=CALTECH101_STD))

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=CALTECH101_MEAN, std=CALTECH101_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=CALTECH101_MEAN, std=CALTECH101_STD))

        train_dataset = Flowers(root=data_root, imb_type='exp', imb_factor=imb_ratio, train=True, download=True,
                                transform=transform_train)
        val_dataset = Flowers(root=data_root, imb_type='exp', imb_factor=imb_ratio, train=False, download=True,
                              transform=val_transform)

    elif dataset == 'fruits':

        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=FRUIT360_MEAN, std=FRUIT360_STD),
            ])
        else:
            for idx, trans in enumerate(transform_train):
                transform_train[idx].transforms.append(transforms.Normalize(mean=FRUIT360_MEAN, std=FRUIT360_STD))

        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=FRUIT360_MEAN, std=FRUIT360_STD)
            ])
        else:
            val_transform.transforms.append(transforms.Normalize(mean=FRUIT360_MEAN, std=FRUIT360_STD))

        train_dataset = Fruits(
            root=data_root,
            train=True,
            transform=transform_train, imb_factor=imb_ratio)
        new_class_idx = train_dataset.get_new_class_idx_sorted()
        val_dataset = Fruits(
            root=data_root,
            train=False,
            transform=val_transform, new_class_idx_sorted=new_class_idx)
    else:
        print('no such dataset')

    if sampler_dic and sampler_dic.get('batch_sampler', False):
        print('Using sampler', sampler_dic['sampler'])
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=sampler_dic['sampler'](train_dataset, **sampler_dic['params']),
            num_workers=num_workers
        )
    elif sampler_dic and meta:
        print('Using sampler', sampler_dic['sampler'])
        print('Sampler parameters: ', sampler_dic['params'])
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler_dic['sampler'](train_dataset, **sampler_dic['params']),
            num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return {'train': train_loader, 'val': val_loader}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fgvc',
                        choices=['inat', 'cub', 'cifar10', 'cifar100', 'fgvc', 'dogs', 'cars', 'flowers'])
    parser.add_argument('--imb_ratio', default=1, type=float)
    parser.add_argument('--data', metavar='DIR', default='/data/')
    args = parser.parse_args()
    train_dataset, val_dataset = get_dataset(args)