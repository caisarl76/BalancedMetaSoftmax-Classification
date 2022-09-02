import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class RandomCycleIter:
    def __init__(self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
        return self.data_list[self.i]


class ClassAwareSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, num_samples_cls=4, ):
        # pdb.set_trace()
        num_classes = len(np.unique(data_source.targets))
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.targets):
            # print(label)
            cls_data_list[label].append(i)

        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:
        if j >= num_samples_cls:
            j = 0
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        i += 1
        j += 1


class LabelAwareSmoothing(nn.Module):
    def __init__(self, cls_num_list, dataset, imb_ratio, shape='concave', power=None, smooth_head=None, smooth_tail=None):
        super(LabelAwareSmoothing, self).__init__()
        if not smooth_tail:
            smooth_head, smooth_tail = get_smooth(dataset=dataset, imb_ratio=imb_ratio)
        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)
        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin(
                (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (
                    n_1 - n_K)
        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(
                1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power(
                (np.array(cls_num_list) - n_K) / (n_1 - n_K), power)
        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        if torch.cuda.is_available():
            self.smooth = self.smooth.cuda()
        self.class_weight = max(torch.Tensor(cls_num_list)) * (1 / torch.Tensor(cls_num_list))
        self.class_weight = self.class_weight.cuda()

    def forward(self, x, target):
        smoothing = self.smooth[target]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss * self.class_weight[target][:, None]
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


class LearnableWeightScaling(nn.Module):
    def __init__(self, num_classes):
        super(LearnableWeightScaling, self).__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        return self.learned_norm * x

def get_smooth(dataset, imb_ratio, _smooth_head=None, _smooth_tail=None):
    if _smooth_head is not None:
        smooth_head = _smooth_head
        smooth_tail = _smooth_tail
    if dataset.lower() == 'cifar10':
        if imb_ratio == 0.1:
            smooth_head = 0.07
            smooth_tail = 0.07
        elif imb_ratio == 0.01:
            smooth_head = 0.05
            smooth_tail = 0.10
    elif dataset.lower() == 'cifar100':
        if imb_ratio == 0.1:
            smooth_head = 0.01
            smooth_tail = 0.19
        elif imb_ratio == 0.01:
            smooth_head = 0.08
            smooth_tail = 0.36
    elif dataset.lower() == 'caltech101':
        smooth_head = 0.01
        smooth_tail = 0.01
    elif dataset.lower() == 'cars':
        smooth_head = 0.0
        smooth_tail = 0.38
    elif dataset.lower() == 'cub':
        smooth_head = 0.04
        smooth_tail = 0.01
    elif dataset.lower() == 'dogs':
        smooth_head = 0.0
        smooth_tail = 0.03
    elif dataset.lower() == 'dtd':
        smooth_head = 0.01
        smooth_tail = 0.39
    elif dataset.lower() == 'flowers':
        smooth_head = 0.04
        smooth_tail = 0.04
    elif dataset.lower() == 'fgvc':
        smooth_head = 0.0
        smooth_tail = 0.39
    elif dataset.lower() == 'inat':
        smooth_head = 0.3
        smooth_tail = 0.0
    elif dataset.lower() == 'imagenet':
        smooth_head = 0.3
        smooth_tail = 0.0
    elif dataset.lower() == 'places':
        smooth_head = 0.4
        smooth_tail = 0.1
    else:
        smooth_head = 0.1
        smooth_tail = 0.0
    print(smooth_head, smooth_tail)
    return smooth_head, smooth_tail
