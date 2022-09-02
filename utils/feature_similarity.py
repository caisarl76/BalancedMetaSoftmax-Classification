import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class similarityLoss(nn.Module):
    def __init__(self, type='custom_cos'):
        super(similarityLoss, self).__init__()
        if type in [None, 'custom_sqrt']:
            print('feat loss type: ', type)
            self.size_average = True
            self.mse = nn.MSELoss(reduction='none')
            self.forward = self.default_forward
        elif 'nnCOS' in type:
            print('feat loss type: ', type)
            self.loss = nn.CosineEmbeddingLoss()
            self.forward = self.cos_forward
        elif 'nnMSE' in type:
            print('feat loss type: ', type)
            self.loss = nn.MSELoss()
            self.forward = self.mse_forward
        else:
            print('wrong loss type: ', type)
            return
        self.sqrt = type is not None and 'sqrt' in type

    def cos_forward(self, s_feat, t_feat, target):
        if self.sqrt:
            s_feat = torch.mul(torch.sign(s_feat), torch.sqrt(torch.abs(s_feat) + 1e-12))
            t_feat = torch.mul(torch.sign(t_feat), torch.sqrt(torch.abs(t_feat) + 1e-12))
        return self.loss(s_feat, t_feat, target)

    def mse_forward(self, s_feat, t_feat, target):
        if self.sqrt:
            s_feat = torch.mul(torch.sign(s_feat), torch.sqrt(torch.abs(s_feat) + 1e-12))
            t_feat = torch.mul(torch.sign(t_feat), torch.sqrt(torch.abs(t_feat) + 1e-12))
        return self.loss(s_feat, t_feat)

    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def default_forward(self, s_feat, t_feat, target=None):
        s_feat_flat = s_feat.view(s_feat.shape[0], -1)
        t_feat_flat = t_feat.view(t_feat.shape[0], -1)
        if self.sqrt:
            s_feat_flat = torch.mul(torch.sign(s_feat_flat), torch.sqrt(torch.abs(s_feat_flat) + 1e-12))
            t_feat_flat = torch.mul(torch.sign(t_feat_flat), torch.sqrt(torch.abs(t_feat_flat) + 1e-12))
        s_feat_flat = nn.functional.normalize(s_feat_flat, dim=1)
        t_feat_flat = nn.functional.normalize(t_feat_flat, dim=1)

        mse = self.mse(s_feat_flat, t_feat_flat)
        mse_sum = mse.sum()
        loss = mse_sum
        if self.size_average:
            loss /= s_feat.size(0)
        return loss


class CosKLD(nn.Module):
    def __init__(self, size_average=True, cls_num_list=None, args=None, sqrt=False):
        super(CosKLD, self).__init__()
        self.size_average = size_average
        self.cls_num_list = cls_num_list
        self.kldiv = nn.KLDivLoss(reduction='sum')
        self.mse = nn.MSELoss(reduction='none')
        self.sqrt = sqrt

    def init_weights(self, init_linear='normal'):
        pass

    def set_cls_num_list(self, cls_num_list):
        self.cls_num_list = cls_num_list

    def forward(self, s_feat, t_feat, class_weight=None, target=None):
        s_feat_flat = s_feat.view(s_feat.shape[0], -1)
        t_feat_flat = t_feat.view(t_feat.shape[0], -1)
        if self.sqrt:
            s_feat_flat = torch.mul(torch.sign(s_feat_flat), torch.sqrt(torch.abs(s_feat_flat) + 1e-12))
            t_feat_flat = torch.mul(torch.sign(t_feat_flat), torch.sqrt(torch.abs(t_feat_flat) + 1e-12))
        s_feat_flat = nn.functional.normalize(s_feat_flat, dim=1)
        t_feat_flat = nn.functional.normalize(t_feat_flat, dim=1)

        mse = self.mse(s_feat_flat, t_feat_flat)
        if class_weight is not None:
            mse = mse * class_weight[target][:, None]
        mse_sum = mse.sum()
        loss = mse_sum
        if self.size_average:
            loss /= s_feat.size(0)
        return loss


class CosSched:
    def __init__(self, args, loader):
        self.current_iter = 0
        self.total_iter = len(loader) * args.epochs
        self.lamda = args.lamda

    def step(self):
        self.current_iter += 1
        lamda = self.lamda[1] - (self.lamda[1] - self.lamda[0]) * (
                (np.cos(np.pi * self.current_iter / self.total_iter)) + 1) / 2
        return lamda


class transfer_conv3(nn.Module):
    def __init__(self, in_feature, out_feature, return_224=False, ch224fix=None, with_96=False):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.return_224 = return_224
        self.with_96 = with_96

        if ch224fix:
            if with_96:
                self.channel = [ch224fix // 2, ch224fix, ch224fix // 2, ch224fix // 2]
            else:
                self.channel = [ch224fix // 2, ch224fix, ch224fix // 2]
        elif with_96:
            self.channel = [out_feature // 4, out_feature // 2, out_feature // 4, out_feature // 4]
        else:
            self.channel = [out_feature // 4, out_feature // 2, out_feature // 4]
        self.conv1 = self._make_layer(in_feature, self.channel[0])
        self.conv2 = self._make_layer(in_feature, self.channel[1])
        self.conv3 = self._make_layer(in_feature, self.channel[2])
        if with_96:
            self.conv4 = self._make_layer(in_feature, self.channel[3])
        # network params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_classifier_input_dim(self):
        return sum(self.channel)

    def _make_layer(self, in_feature, out_feature):
        layers = nn.Sequential()
        layers.add_module('conv', nn.Conv2d(in_feature, out_feature, kernel_size=1, bias=False))
        layers.add_module('bn', nn.BatchNorm2d(out_feature))
        layers.add_module('act', nn.LeakyReLU())
        return layers

    def forward(self, features):
        feat1 = F.interpolate(self.conv1(features[0]), size=(7, 7))
        feat2 = F.interpolate(self.conv2(features[1]), size=(7, 7))
        feat3 = F.interpolate(self.conv3(features[2]), size=(7, 7))
        if self.with_96:
            feat4 = F.interpolate(self.conv3(features[3]), size=(7, 7))
            add_feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        else:
            add_feats = torch.cat([feat1, feat2, feat3], 1)
        if self.return_224:
            return add_feats, feat2
        else:
            return add_feats


class mlp(nn.Module):
    def __init__(self, features=[3840, 1280], feat_cat=True):
        super().__init__()
        self.features = features
        self.depth = len(features) - 1
        self.layer = self._make_layer()
        self.feat_cat = feat_cat

    def _make_layer(self):
        layers = nn.Sequential()
        layers.add_module('layer_0', nn.Linear(self.features[0], self.features[1]))
        for i in range(1, self.depth):
            layers.add_module('relu_%d' % i, nn.ReLU())
            layers.add_module('layer_%d' % i, nn.Linear(self.features[i], self.features[i + 1]))
        return layers

    def forward(self, vectors):
        if self.feat_cat:
            vectors = torch.cat(vectors, dim=1)
        return self.layer(vectors)
    # need to implement return 224 features


class single_mlp(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.mlp = self._make_layer(depth=depth, out_features=1280)

    def _make_layer(self, depth=3, out_features=1280):
        layers = nn.Sequential()
        for i in range(depth - 1):
            layers.add_module('layer_%d' % i, nn.Linear(1280, 1280))
            layers.add_module('relu_%d' % i, nn.ReLU())
        layers.add_module('layer_%d' % (depth - 1), nn.Linear(1280, out_features))
        return layers

    def forward(self, x):
        return self.mlp(x)


class attention_multi(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        if depth == 2:
            in_feature_list = [3840, 1280]
        elif depth == 4:
            in_feature_list = [3840, 3840, 2560, 1280]
        elif depth == 6:
            in_feature_list = [3840, 3840, 3840, 3840, 2560, 1280]
        else:
            print('Wrong depth value', depth)
            assert False
        self.mlp = self._make_layer(in_feature_list=in_feature_list)

    def _make_layer(self, in_feature_list):
        layers = nn.Sequential()
        depth = len(in_feature_list)
        for i in range(depth - 1):
            layers.add_module('layer_%d' % i,
                              nn.Linear(in_features=in_feature_list[i], out_features=in_feature_list[i + 1]))
            layers.add_module('relu_%d' % i, nn.ReLU())
        layers.add_module('layer_%d' % depth, nn.Linear(in_features=in_feature_list[-1], out_features=3))
        return layers

    def forward(self, vectors):
        vector = torch.cat(vectors, dim=1)
        output = self.mlp(vector)
        factors = F.softmax(output, dim=1)
        final_vector = (factors.T[0] * vectors[0].T).T + \
                       (factors.T[1] * vectors[1].T).T + \
                       (factors.T[2] * vectors[2].T).T
        return final_vector


class transfer_conv(nn.Module):
    def __init__(self, in_feature, out_feature, depthwise=False, filter_size=1):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.Connectors = nn.Sequential()
        if depthwise:
            self.Connectors.add_module('dw', nn.Conv2d(out_feature, out_feature, kernel_size=filter_size, stride=1,
                                                       padding=0, bias=False, groups=out_feature))
            self.Connectors.add_module('bn_dw', nn.BatchNorm2d(out_feature))
            self.Connectors.add_module('act_dw', nn.LeakyReLU())
        else:
            self.Connectors.add_module('conv', nn.Conv2d(in_feature, out_feature, kernel_size=filter_size, stride=1,
                                                         padding=filter_size - 1, bias=False))
            self.Connectors.add_module('bn', nn.BatchNorm2d(out_feature))
            self.Connectors.add_module('act', nn.LeakyReLU())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, student):
        student = self.Connectors(student)
        return student


class transfer_mlp(nn.Module):
    def __init__(self, version=2, depth=3, is_small=False, channel_fix=False, multi3=False, is_tres=False,
                 in_feature=1280, return_224=False,
                 ch224fix=None):
        super().__init__()
        self.version = version
        self.return_224 = return_224

        if multi3:
            self.cat_dim = 0
        else:
            self.cat_dim = 1
        if is_small:
            self.forward = self.small_forward
        else:
            self.forward = self.large_forward

        self.in_feature = in_feature
        self.channel_fix = channel_fix
        if self.channel_fix:
            channel = [in_feature, in_feature, in_feature]
        else:
            channel = [in_feature // 4, in_feature // 2, in_feature // 4]

        if is_tres:
            depth = 3
            channel = [2048 // 4, 2048 // 2, 2048 // 4]

        self.mlp1 = self._make_layer(in_features=in_feature, depth=depth, out_features=(channel[0]))
        self.mlp2 = self._make_layer(in_features=in_feature, depth=depth, out_features=(channel[1]))
        self.mlp3 = self._make_layer(in_features=in_feature, depth=depth, out_features=(channel[2]))

    def _make_layer(self, in_features=1280, depth=3, out_features=1280):
        layers = nn.Sequential()
        for i in range(depth - 1):
            layers.add_module('layer_%d' % i, nn.Linear(in_features, in_features))
            layers.add_module('relu_%d' % i, nn.ReLU())
        layers.add_module('layer_%d' % (depth - 1), nn.Linear(in_features, out_features))
        return layers

    def small_forward(self, inputs):
        if self.mlp2:
            feat1 = self.mlp2(inputs[0])
        else:
            feat1 = inputs[0]
        feat2 = self.mlp1(inputs[1])
        feat3 = self.mlp3(inputs[2])
        feat = torch.cat([feat1, feat2, feat3], dim=self.cat_dim)
        return feat

    def large_forward(self, inputs):
        if type(inputs) != list:
            feat = self.mlp2(inputs)
            return feat
        feat1 = self.mlp1(inputs[0])
        feat3 = self.mlp3(inputs[2])
        if self.mlp2:
            feat2 = self.mlp2(inputs[1])
        else:
            feat2 = inputs[1]

        feat = torch.cat([feat1, feat2, feat3], dim=self.cat_dim)
        if self.return_224:
            return feat, feat2
        else:
            return feat


class conv_three(nn.Module):
    def __init__(self, in_features, out_features, depth, ratio_fix=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.depth = depth

        if ratio_fix:
            channel = [out_features // 4, out_features // 2, out_features // 4]
        else:
            channel = [out_features, out_features, out_features]
        self.mlp = self._make_layer(channel)

    def _make_layer(self, channel):
        mlps = []
        for i in range(3):
            layers = nn.Sequential()
            for dpt in range(self.depth - 1):
                layers.add_module('conv%d' % (dpt), nn.Conv2d(self.in_features, self.in_features, stride=1, bias=False))
                layers.add_module('bn%d' % (dpt), nn.BatchNorm2d(self.in_features))
                layers.add_module('act%d' % (dpt), nn.ReLU())
            layers.add_module('conv%d' % (self.depth - 1),
                              nn.Conv2d(self.in_features, self.out_features, stride=1, bias=False))
            layers.add_module('bn%d' % (self.depth - 1), nn.BatchNorm2d(self.out_features))
            layers.add_module('act%d' % (self.depth - 1), nn.ReLU())
            mlps.append(layers)
        return mlps
