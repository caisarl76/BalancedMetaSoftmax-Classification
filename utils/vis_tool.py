import os
from collections import OrderedDict

import seaborn as sns
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

import tqdm
from sklearn.manifold import TSNE

from .feature_similarity import CosKLD


def plot_vecs_n_labels(v, labels, fname, num_classes):
    fig = plt.figure(figsize=(30, 30))
    plt.axis('off')
    sns.set_style('darkgrid')
    # sns.scatterplot(v[:, 0], v[:, 1], hue=labels, legend='full', palette=sns.color_palette("bright", as_cmap=True))
    plt.scatter(v[:, 0], v[:, 1], c=labels)
    plt.legend()
    plt.savefig(fname)
    print(f'saved on {fname}')


def get_tsne(args, loader, enc, fc=None, file_name=''):
    tsne = TSNE()
    for i, (x, y) in enumerate(tqdm.tqdm(loader)):
        x = x.cuda()
        with torch.no_grad():
            _, pred = enc(x)
            if fc is not None:
                pred = fc(pred)
                pred = torch.softmax(pred, dim=1)
            if i == 0:
                all_outputs = pred
                all_targets = y
            else:
                all_outputs = torch.cat((all_outputs, pred))
                all_targets = torch.cat((all_targets, y))
    pred_tsne = tsne.fit_transform(all_outputs.cpu().samples)
    plot_vecs_n_labels(pred_tsne, all_targets, os.path.join(args.root_model, f'{file_name}_tsne.pdf'), args.num_classes)


class AccumulateMeter(object):
    def __init__(self, shape=1):
        self.reset(shape)

    def reset(self, shape):
        self.val = np.zeros(shape)
        self.avg = np.zeros(shape)
        self.sum = np.zeros(shape)
        self.count = np.zeros(shape)

    def update(self, val, n=1):
        self.val = val
        self.sum += np.multiply(val, n)
        self.count += n
        self.avg = self.sum / self.count


def get_layerwise_gradients(args, student_enc, feat_converter, teacher_model, loader, wanted_class=0):
    feat_loss = CosKLD(args=args)
    if args.classwise_bound:
        class_weight = max(torch.Tensor(args.cls_num_list)) * (1 / torch.Tensor(args.cls_num_list))
        class_weight = class_weight.cuda()
    else:
        class_weight = None

    grad_collector = OrderedDict()
    for n, m in student_enc.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            grad_collector[n] = AccumulateMeter(m.weight.shape)

    for i, (x, y) in enumerate(tqdm.tqdm(loader)):
        wanted_idx = (y == wanted_class).nonzero(as_tuple=True)[0]
        x = x[wanted_idx].cuda()
        y = y[wanted_idx].cuda()
        print(y)
        print(len(y))
        if len(y) == 0:
            continue
        s_feat, pred = student_enc(x)
        # pred = student_fc(pred)
        s_feat = feat_converter(s_feat)
        with torch.no_grad():
            t_feat, t_out = teacher_model(x)
            # t_out = lws_model(t_out)

        student_enc.zero_grad()
        loss = feat_loss(s_feat=s_feat, t_feat=t_feat, class_weight=class_weight, target=y)
        loss.backward()

        for n, m in student_enc.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                grad_collector[n].update(m.weight.grad.detach().cpu().numpy(), y.size(0))

    return grad_collector

def get_image_confidence(args, student_enc, student_fc, loader):
    is_first = True
    with torch.no_grad():
        for x, y in loader:
            wanted_idx = torch.eq(y, args.wanted_class).nonzero(as_tuple=True)[0]
            if len(wanted_idx) == 0:
                continue
            img = x[wanted_idx].cuda()
            # y = y[wanted_idx].cuda()
            _, feat = student_enc(img)
            output = student_fc(feat)
            output = torch.softmax(output, dim=1)

            # print(output)
            # print(output_idx)
            if is_first:
                images = img
                indices = output
                is_first = False
            else:
                images = torch.cat((images, img))
                indices = torch.cat((indices, output))
    indices = torch.argsort(indices[:, args.wanted_class], descending=True, dim=0)
    # print(indices)
    # print(images.shape)
    # print(indices.shape)
    plt.figure()
    plt.clf()
    plt.axis('off')
    plt.imshow(torchvision.utils.make_grid(images[indices].cpu(), normalize=True).permute(1, 2, 0))

    os.makedirs(os.path.join(args.root_model, args.file_name), exist_ok=True)
    plt.savefig(os.path.join(args.root_model, args.file_name, f'{args.wanted_class}_confidence.pdf'))