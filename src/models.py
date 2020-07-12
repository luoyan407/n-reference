import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data.sampler import Sampler

import numpy as np
import time
import random
import scipy
import cv2
import copy
import math
import collections
import scipy.io
import scipy.ndimage
import scipy.misc
import copy
import quadprog

from PIL import Image, ImageOps

import os, sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src')
sys.path.append(srcFolder)
from utils import *

map_size = (30, 40)
nnupsample = nn.Upsample
meps = np.finfo(float).eps

def deactivate_batchnorm(m):
    if isinstance(m, nn.modules.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

def set_bn_eval(m):
    #if isinstance(m, nn.modules.batchnorm._BatchNorm):
    if isinstance(m, nn.modules.BatchNorm2d):
        m.eval()

def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()

class FixedRandomSampler(Sampler):
    def __init__(self, num_samples, num_epochs, save_shuffles):
        shuffles = {}
        if not os.path.isfile(save_shuffles):
            f_sequences = list(range(num_samples))
            for i in range(num_epochs):
                np.random.shuffle(f_sequences)
                shuffles[i] = f_sequences.copy()
            torch.save(shuffles, save_shuffles)
        else:
            shuffles = torch.load(save_shuffles)

        self.shuffles = shuffles
        self.count = 0
    def __iter__(self):
        self.count += 1
        return iter(self.shuffles[self.count-1])
    def __len__(self):
        return len(self.shuffles[0])

class ExtendedRandomSampler(Sampler):
    def __init__(self, actual_size, expected_size):
        self.actual_size = actual_size
        self.expected_size = expected_size

    def __iter__(self):
        indices = np.random.randint(self.actual_size, size=self.expected_size ).tolist()
        return iter(indices)

    def __len__(self):
        return self.expected_size

def g_filter(shape =(200,200), sigma=60):
    """
    Using Gaussian filter to generate center bias
    """
    x, y = [edge /2 for edge in shape]
    x = int(x)
    y = int(y)
    grid = np.array([[((i**2+j**2)/(2.0*sigma**2)) for i in range(-x, x)] for j in range(-y, y)])
    g_filter = np.exp(-grid)/(2*np.pi*sigma**2)
    g_filter /= np.sum(g_filter)
    return g_filter

def to_np(x):
    return x.data.cpu().numpy()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val / n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=(1,1), padding=(2,2)):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=(1,1), padding=(1,1)):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, dilation=1, padding=0):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class DIN_encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DIN_encoder, self).__init__()
        self.conv0 = conv1x1(in_channels, out_channels)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(out_channels, out_channels, dilation=(4,4), padding=(4,4))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, dilation=(8,8), padding=(8,8))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(out_channels, out_channels, dilation=(16,16), padding=(16,16))
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)

        x1 = self.conv1(x)
        x1 = self.relu1(x1)

        x2 = self.conv2(x)
        x2 = self.relu2(x2)

        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        out = x1+x2+x3

        return out

class DIN_decoder(nn.Module):
    def __init__(self, in_channels=256):
        super(DIN_decoder, self).__init__()
        self.conv1 = conv3x3(in_channels, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels, in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x

class Saliency_DIN(nn.Module):
    def __init__(self, 
                modelType, 
                modelzoo, 
                pretrained=True,
                n_output=256):
        super(Saliency_DIN, self).__init__()
        net = modelzoo['resnet50'](pretrained=pretrained)

        net.layer3[0].conv2.stride = (1,1)
        net.layer3[0].downsample.__getitem__(0).stride = (1,1)
        net.layer4[0].conv2.stride = (1,1)
        net.layer4[0].downsample.__getitem__(0).stride = (1,1)
        for layer in net.layer3.modules():
            if isinstance(layer, nn.modules.conv.Conv2d):
                if layer.kernel_size == (3,3):
                    layer.dilation = (2,2)
                    layer.padding = (2,2)
        for layer in net.layer4.modules():
            if isinstance(layer, nn.modules.conv.Conv2d):
                if layer.kernel_size == (3,3):
                    layer.dilation = (4,4)
                    layer.padding = (4,4)

        features = list(net.children())[:-2]
        features.append(DIN_encoder(2048,n_output))
        features.append(DIN_decoder(n_output))
        features.append(conv3x3(n_output, 1, padding=1))
        features.append(nn.Sigmoid())
        self.net = nn.Sequential(*features)

    def forward(self, x):
        x = self.net(x)

        return x

    def get_param_groups(self):
        
        return [{'params': self.net.parameters()}]

def split_model(model):
    layers = list(model.children())
    sublayers = list(layers[1].children())
    model_body = nn.Sequential( layers[0],*sublayers[:2] )
    model_head = nn.Sequential( *sublayers[2:] )
    return model_body, model_head

def split_model_din(model, split_layer=1):
    layers = list(model.children())
    layers = list(layers[0].children())
    idx = -2+1-split_layer
    model_body = nn.Sequential( *layers[:idx] )
    model_head = nn.Sequential( *layers[idx:] )
    return model_body, model_head

def store_grad(pp, grad_buffer, grad_dims):
    """
        This funtion is built on store_grad in https://github.com/facebookresearch/GradientEpisodicMemory
    """
    # store the gradients
    grad_buffer.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grad_buffer[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1

def overwrite_grad(pp, newgrad, grad_dims):
    """
        This funtion is built on store_grad in https://github.com/facebookresearch/GradientEpisodicMemory
    """
    # overwrite the gradients
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

class Referencer(nn.Module):
    def __init__(self, 
                subnet,
                th=0.0,
                lr=1.0,
                lamb=1.0):
        super(Referencer, self).__init__()
        self.subnet = subnet
        # th: If the cosine similarity is less than threshold, apply the reference process
        # lr: As using sgd to optimize the cosine similarity, this lr is used to update the gradient of the gradient
        # lamb: The regularization weight
        self.th = th
        self.lr = lr
        self.lamb = lamb

        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_buffer = torch.Tensor(sum(self.grad_dims))
        self.grad_ref = torch.Tensor(sum(self.grad_dims))

    def forward(self, x):
        x = self.subnet(x)
        
        return x

    def refer_and_learn(self, x, y, x_ref, y_ref, 
                        criterion, bodynet_optimizer, headnet_optimizer):
        # First compute the gradients w.r.t. the reference samples
        bodynet_optimizer.zero_grad()
        headnet_optimizer.zero_grad()
        x_ref_out = self.forward(x_ref)
        if x_ref_out.shape[2] != y_ref.shape[2] or x_ref_out.shape[3] != y_ref.shape[3]:
            x_ref_out = F.interpolate(x_ref_out,size=y_ref.size()[2:], mode='bilinear', align_corners=True)
        loss_ref = criterion(x_ref_out, y_ref)
        loss_ref.backward()
        store_grad(self.parameters, self.grad_ref, self.grad_dims)

        # Second compute the gradients w.r.t. the training samples
        bodynet_optimizer.zero_grad()
        headnet_optimizer.zero_grad()
        x_out = self.forward(x)
        if x_out.shape[2] != y.shape[2] or x_out.shape[3] != y.shape[3]:
            x_out = F.interpolate(x_out,size=y.size()[2:], mode='bilinear', align_corners=True)
        loss = criterion(x_out, y)
        loss.backward()
        store_grad(self.parameters, self.grad_buffer, self.grad_dims)


        tmp = F.cosine_similarity(self.grad_buffer, self.grad_ref, dim=0)
        cos_sim = [tmp.item(), tmp.item()] # cosine similarities [before, after] the reference process
        if cos_sim[0] < self.th:
            # Optimize the cosine similarity between the reference gradient and the training gradient
            tmp_buffer = self.grad_buffer.clone().detach().requires_grad_(True)
            cos_loss = F.cosine_similarity(tmp_buffer, self.grad_ref, dim=0) - self.lamb*torch.norm(tmp_buffer, p=2, dim=0)**2
            buffer_grad = torch.autograd.grad(outputs=cos_loss, inputs=tmp_buffer)[0]
            self.grad_buffer = self.grad_buffer + self.lr * buffer_grad

            # Write it back to the variable
            overwrite_grad(self.parameters, self.grad_buffer, self.grad_dims)

            # Re-compute the cosine similarity for statistics purposes
            tmp = F.cosine_similarity(self.grad_buffer, self.grad_ref, dim=0)
            cos_sim[1] = tmp.item()

        # Update the parameters
        headnet_optimizer.step()
        bodynet_optimizer.step()

        return x_out, loss, cos_sim

class Saliency_ResNet50(nn.Module):
    def __init__(self, 
                modelType, 
                modelzoo, 
                pretrained=True):
        super(Saliency_ResNet50, self).__init__()
        net = modelzoo['resnet50'](pretrained=pretrained)
        net.layer3[0].conv2.stride = (1,1)
        net.layer3[0].downsample.__getitem__(0).stride = (1,1)
        net.layer4[0].conv2.stride = (1,1)
        net.layer4[0].downsample.__getitem__(0).stride = (1,1)
        for layer in net.layer3.modules():
            if isinstance(layer, nn.modules.conv.Conv2d):
                if layer.kernel_size == (3,3):
                    layer.dilation = (2,2)
                    layer.padding = (2,2)
        for layer in net.layer4.modules():
            if isinstance(layer, nn.modules.conv.Conv2d):
                if layer.kernel_size == (3,3):
                    layer.dilation = (4,4)
                    layer.padding = (4,4)
        features = list(net.children())[:-2]
        features.append(conv3x3(2048, 1))

        features.append(nn.Sigmoid())
        self.net = nn.Sequential(*features)

    def forward(self, x):
        x = self.net(x)

        return x

    def get_param_groups(self):
        
        return [{'params': self.net.parameters()}]