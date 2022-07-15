import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

class ImageNetLT(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        self.cls_num_list = [np.sum(np.array(self.targets) == i) for i in range(1000)]
        self.cls_num_list = list([int(x) for x in self.cls_num_list])

    def get_cls_num_list(self):
        if not os.path.exists('cls_freq'):
            os.makedirs('cls_freq')
        freq_path = os.path.join('cls_freq', 'imagenet.json')
        with open(freq_path, 'w') as fd:
            json.dump(self.cls_num_list, fd)
        return self.cls_num_list

    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.targets[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label , index
