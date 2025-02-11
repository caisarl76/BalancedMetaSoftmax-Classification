import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

class Flowers(Dataset):
    def __init__(self, root, train=True, download=False, transform=None, rand_number=0, imb_factor=1, imb_type='exp', random_seed=False):
        np.random.seed(rand_number)
        self.random_seed = random_seed
        root = os.path.join(root, 'flowers')
        if train:
            excel_file = os.path.join(root, 'train.txt')
        else:
            excel_file = os.path.join(root, 'valid.txt')

        self.samples = pd.read_csv(excel_file, delimiter=' ')
        self.root_dir = root
        self.transform = transform
        self.targets = self.samples['TARGET'].array
        self.classes = np.unique(self.targets)
        self.cls_num = len(self.classes)

        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets, dtype=np.int64)

        num_in_class = []
        for class_idx in np.unique(self.targets):
            num_in_class.append(len(np.where(self.targets == class_idx)[0]))
        self.num_in_class = num_in_class
        if train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
        self.many_shot_idx = 36
        self.median_shot_idx = 72

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.samples) / cls_num
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

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        classes = np.unique(self.targets)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(self.targets == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            self.num_per_cls_dict[the_class] = len(selec_idx)
            new_data.append(self.samples[selec_idx])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.samples = new_data
        self.targets = new_targets
        self.labels = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.samples[index, 0])
        y_label = torch.tensor(self.samples[index, 1]).long()
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, y_label, index
