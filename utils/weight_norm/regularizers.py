import numpy as np
import torch
import torch.nn as nn
import math


# The classes below wrap core functions to impose weight regurlarization constraints in training or finetuning a network.

class MaxNorm_via_PGD():
    # learning a max-norm constrainted network via projected gradient descent (PGD)
    def __init__(self, model_type='efficient', thresh=1.0, LpNorm=2, tau=1):
        self.model_type = model_type
        self.thresh = thresh
        self.LpNorm = LpNorm
        self.tau = tau
        self.perLayerThresh = []

    def setPerLayerThresh(self, model, mlp=None):
        # set per-layer thresholds
        is_module = True if hasattr(model, "module") else False
        if self.model_type == 'efficient':
            if is_module:
                target_layer = [model.module.classifier.weight, model.module.classifier.bias]
            else:
                target_layer = [model.classifier.weight, model.classifier.bias]
        elif self.model_type == 'resnet152':
            if is_module:
                target_layer = [model.module.fc.weight, model.module.fc.bias]
            else:
                target_layer = [model.fc.weight, model.fc.bias]
        else:
            if is_module:
                target_layer = [model.module.head.fc.weight, model.module.head.fc.bias]
            else:
                target_layer = [model.head.fc.weight, model.head.fc.bias]
        if mlp is not None:
            target_layer+= list(mlp.parameters())

        self.perLayerThresh = []

        for curLayer in target_layer:  # here we only apply MaxNorm over the last two layers
            curparam = curLayer.data
            if len(curparam.shape) <= 1:
                self.perLayerThresh.append(float('inf'))
                continue
            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1).detach().unsqueeze(-1)
            curLayerThresh = neuronNorm_curparam.min() + self.thresh * (
                    neuronNorm_curparam.max() - neuronNorm_curparam.min())
            self.perLayerThresh.append(curLayerThresh)
        return target_layer

    def PGD(self, model, mlp=None):
        is_module = True if hasattr(model, "module") else False
        if self.model_type == 'efficient':
            if is_module:
                target_layer = [model.module.classifier.weight, model.module.classifier.bias]
            else:
                target_layer = [model.classifier.weight, model.classifier.bias]
        elif self.model_type == 'resnet152':
            if is_module:
                target_layer = [model.module.fc.weight, model.module.fc.bias]
            else:
                target_layer = [model.fc.weight, model.fc.bias]
        else:
            if is_module:
                target_layer = [model.module.head.fc.weight, model.module.head.fc.bias]
            else:
                target_layer = [model.head.fc.weight, model.head.fc.bias]
        if mlp is not None:
            target_layer+= list(mlp.parameters())
        if len(self.perLayerThresh) == 0:
            self.setPerLayerThresh(model, mlp)

        for i, curLayer in enumerate(target_layer):  # here we only apply MaxNorm over the last two layers
            curparam = curLayer.data

            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = (
                    torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1) ** self.tau).detach().unsqueeze(-1)
            scalingVect = torch.ones_like(curparam)
            curLayerThresh = self.perLayerThresh[i]

            idx = neuronNorm_curparam > curLayerThresh
            idx = idx.squeeze()
            tmp = curLayerThresh / (neuronNorm_curparam[idx].squeeze()) ** (self.tau)
            for _ in range(len(scalingVect.shape) - 1):
                tmp = tmp.unsqueeze(-1)

            scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
            curparam[idx] = scalingVect[idx] * curparam[idx]


class Normalizer():
    def __init__(self, LpNorm=2, tau=1):
        self.LpNorm = LpNorm
        self.tau = tau

    def apply_on(self, model):  # this method applies tau-normalization on the classifier layer

        for curLayer in [model.encoder.fc.weight]:  # change to last layer: Done
            curparam = curLayer.data

            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = (
                    torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1) ** self.tau).detach().unsqueeze(-1)
            scalingVect = torch.ones_like(curparam)

            idx = neuronNorm_curparam == neuronNorm_curparam
            idx = idx.squeeze()
            tmp = 1 / (neuronNorm_curparam[idx].squeeze())
            for _ in range(len(scalingVect.shape) - 1):
                tmp = tmp.unsqueeze(-1)

            scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
            curparam[idx] = scalingVect[idx] * curparam[idx]
