import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
import importlib
import math
import shutil

import numpy as np


def mixup_data(x, y, alpha=1.0, use_cuda=True, seed=100):
    random.seed(seed)
    torch.random.manual_seed(seed)
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


class FeatCoef:
    def __init__(self, epochs):
        self.epochs = epochs
        self.cur_epoch = 0

    def get_coef(self):
        coefs = 1e-5 + (1 / 2) * (1 - 1e-5) * (1 + np.cos((self.cur_epoch / self.epochs) * np.pi))
        self.cur_epoch += 1
        return coefs


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(''
                        ''
                        ''.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a.cuda()) + (1 - lam) * criterion(pred, y_b.cuda())


def kd_kl_loss(o_student, o_teacher, T=2, w=1, gamma=1, cls_num_list=None, t_sqaure=False, option='student',
               reduction='mean'):
    if cls_num_list:
        cls_num_list = torch.Tensor(cls_num_list).view(1, len(cls_num_list))
        weight = cls_num_list / cls_num_list.sum()
        weight = weight.to(torch.device('cuda'))
        o_student = o_student + torch.log(weight + 1e-9) * gamma
        if option == 'teacher_student':
            o_teacher = o_teacher + torch.log(weight + 1e-9) * gamma
    if t_sqaure:
        kl_loss = nn.KLDivLoss(reduction=reduction)(F.log_softmax(o_student / T, dim=1),
                                                    F.softmax(o_teacher / T, dim=1)) * w * (T ** 2)
    else:
        kl_loss = nn.KLDivLoss(reduction=reduction)(F.log_softmax(o_student / T, dim=1),
                                                    F.softmax(o_teacher / T, dim=1)) * w
    return kl_loss


def kd_kl_loss_reverse(o_student, o_teacher, T=2, w=1, gamma=1, cls_num_list=None):
    if cls_num_list:
        cls_num_list = torch.Tensor(cls_num_list).view(1, len(cls_num_list))
        weight = cls_num_list / cls_num_list.sum()
        weight = weight.to(torch.device('cuda'))
        o_student += torch.log(weight + 1e-9) * gamma
        o_teacher += torch.log(weight + 1e-9) * gamma
    kl_loss = nn.KLDivLoss()(F.log_softmax(o_teacher / T, dim=1), F.softmax(o_student / T, dim=1)) * w
    return kl_loss


def feature_loss_function(fea, target_fea):
    loss = ((fea - target_fea) ** 2).float()
    return torch.abs(loss).mean()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, args):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        open(args.root_path + "/" + "log.log", "a+").write('\t'.join(entries) + "\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1) / (args.epochs - args.warmup_epochs + 1)))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.cpu().topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.cpu().view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
