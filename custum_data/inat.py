from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np


class INaturalist(Dataset):
    num_classes = 8142
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        self.cls_num_list = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]


    def get_cls_num_list(self):
        return self.cls_num_list

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            if type(self.transform) == list:
                sample1 = self.transform[0](sample)
                sample2 = self.transform[1](sample)
                return [sample1, sample2], label
            else:
                sample = self.transform(sample)
                return sample, label, index
