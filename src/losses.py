import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

meps = np.finfo(float).eps
epsilon = 1e-7 #regularization value in Keras

upsample_mode = 'bilinear'

class loss_wrapper():
    """docstring for nn.module"""
    def __init__(self, func):
        super(loss_wrapper, self)
        self.func = func

    def __call__(self, x, y):
        loss = self.func(x, y)
        return loss
    
    def __repr__(self):
        str = '{}'.format(self.func.__name__)
        return str

def spatial_cross_entropy(input, gt):
    if (input.shape[2] != gt.shape[2] and 
        input.shape[3] != gt.shape[3]):
        input = F.interpolate(input, 
            size=tuple(gt.size()[2:]), mode=upsample_mode)

    input_sum = torch.sum(torch.sum(input,dim=3,keepdim=True),dim=2,keepdim=True).expand_as(input)
    input_nrm = input/input_sum
    celoss = -gt*torch.log(input_nrm+meps)-(1-gt)*(torch.log(1-input_nrm+meps))
    celoss = torch.mean(celoss)
    return celoss

def approach_1storder(y_pred, y, eps=1e-7):
    y_sum = y.sum(dim=[1, 2, 3]).view(-1,1,1,1)
    P = y / (eps + y_sum)
    y_pred_sum = y_pred.sum(dim=[1, 2, 3]).view(-1,1,1,1)
    Q = y_pred / (eps + y_pred_sum)
    loss = (Q-P)*torch.log(Q/(P+eps)+eps)
    loss = torch.sum(loss)
    return loss

def normalized_l1(y_pred, y, eps=1e-7, p=1):
    y_min = y.min(dim=3)[0].min(dim=2)[0].min(dim=1)[0].view(-1,1,1,1)
    P = y - y_min
    y_max = P.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0].view(-1,1,1,1)
    P = P / (eps + y_max)

    y_pred_min = y_pred.min(dim=3)[0].min(dim=2)[0].min(dim=1)[0].view(-1,1,1,1)
    Q = y_pred - y_pred_min
    y_pred_max = Q.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0].view(-1,1,1,1)
    Q = Q / (eps + y_pred_max)

    tv = torch.norm(P - Q, p=p)/(y.shape[2]*y.shape[3])
    return tv

def CEdist(y_pred, y, eps=1e-7):
    P = y / (eps + y.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    Q = y_pred / (eps + y_pred.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    tmp = -P*torch.log(Q+eps)
    tv = torch.sum( tmp )*0.05
    return tv

def TVdist_reverse(y_pred, y, eps=1e-7, scalar=.5):
    y = y.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0].view(-1,1,1,1) - y
    P = y / (eps + y.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    Q = y_pred / (eps + y_pred.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    loss = torch.sum( torch.abs(P - Q) ) * scalar
    return loss

def TVdist_sep(y_pred, y, eps=1e-7, alpha=(0,1), scale=1):
    y_sep = y - y.min()
    y_sep = y_sep / y_sep.max()
    y_sep = y_sep - alpha[0]
    y_sep = (y_sep >= 0).type_as(y_sep) * y_sep
    y_sep[y_sep >= (alpha[1] - alpha[0])] = alpha[1] - alpha[0]
    y_sep = y_sep - y_sep.min()
    y_sep = y_sep / y_sep.max()

    P = y_sep / (eps + y_sep.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    Q = y_pred / (eps + y_pred.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    tv = torch.sum( torch.abs(P - Q) ) * scale
    return tv

def TVdist_spatial(y_pred, y, eps=1e-7, coef=0.5):
    P = y / (eps + y.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    Q = y_pred / (eps + y_pred.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    loss = torch.abs(P - Q) * coef
    return loss

def TVdist_sum(y_pred, y, func=TVdist_spatial, eps=1e-7, coef=0.5, mode='train'):
    loss = func(y_pred, y, eps=eps, coef=coef)
    if mode == 'train':
        discard_idx = torch.LongTensor(1).random_(loss.shape[0])[0].item()
        if discard_idx == 0:
            loss = loss[discard_idx+1:,:,:,:]
        elif discard_idx == loss.shape[0]-1:
            loss = loss[:discard_idx,:,:,:]
        else:
            loss = torch.cat((loss[:discard_idx,:,:,:],loss[discard_idx+1:,:,:,:]), dim=0)
    return torch.sum(loss)

def TVdist_mask(y_pred, y, func=TVdist_spatial, eps=1e-7, coef=0.5, mode='train', mask=None):
    loss = func(y_pred, y, eps=eps, coef=coef)
    if mode == 'train':
        if mask is not None:
            loss = mask * loss
    return torch.sum(loss)

class TVdist_wrapper():
    """docstring for nn.module"""
    def __init__(self, eps=1e-7, alpha=(0,1), scale=1, func=TVdist_sep):
        super(TVdist_wrapper, self)
        self.func = func
        self.eps = eps
        self.alpha = alpha
        self.scale = scale

    def __call__(self, x, y):
        loss = self.func(x, y, self.eps, self.alpha, self.scale)
        return loss
    
    def __repr__(self):
        str = '{}'.format(self.func.__name__)
        return str

def TVdist_gt(y_pred, y, eps=1e-7):
    P = y / (eps + y.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    Q = y_pred / (eps + y_pred.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    tv = torch.sum( (P>(P.max().item()*0.5)).type_as(P)*torch.abs(P - Q) ) *1000
    return tv

def TVdist(y_pred, y, eps=1e-7):
    P = y / (eps + y.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    Q = y_pred / (eps + y_pred.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    tv = torch.sum( torch.abs(P - Q) )*0.5
    return tv

def TVdist_c(y_pred, y, 
                eps=1e-7, threshold=0.8, pos_coef=2.0, neg_coef=.8):
    P = y / (eps + y.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    Q = y_pred / (eps + y_pred.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    l1_norm = torch.abs(P - Q)
    cond_P = P >= threshold*P.max().item()
    cond_P = cond_P.type_as(P)
    cond_pos = cond_P * pos_coef
    cond_neg = (1-cond_P) * neg_coef
    l1_norm = l1_norm*(cond_pos+cond_neg)

    tv = torch.sum( l1_norm )*0.5
    return tv

def TVdist_n(y_pred, y, 
                eps=1e-7, base=1.0):
    P = y / (eps + y.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    Q = y_pred / (eps + y_pred.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    l1_norm = torch.abs(P - Q)

    l1_norm = (base+2*y/y.max())*l1_norm

    tv = torch.sum( l1_norm )*0.5
    return tv

class Pull_and_push():
    """docstring for nn.module"""
    def __init__(self, eps=1e-7, lossfunc=None):
        super(Pull_and_push, self)
        self.lossfunc = lossfunc
        self.eps = eps

    def __call__(self, y_2nd, y_1st, y):
        loss_pull = self.lossfunc(y_2nd, y, eps=self.eps)
        loss_push = self.lossfunc(y_2nd, y_1st, eps=self.eps)
        loss = loss_pull - loss_push
        return loss
    
    def __repr__(self):
        str = '{}'.format(self.lossfunc.__name__)
        return str

def TVdist_norm(y_pred, y, eps=1e-7, p=1., t=.5):
    P = y / (eps + y.sum(dim=[1, 2, 3]).view(-1,1,1,1).abs())
    Q = y_pred / (eps + y_pred.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    tv = torch.sum( torch.abs(P - Q) )**(1./p)*t
    return tv

def TVdist_2scale(y_pred, y, eps=1e-7):
    y_pred_clone = y_pred.clone()
    y_clone = y.clone()
    y_pred_clone = F.interpolate(y_pred_clone, size=tuple([y_pred_clone.shape[2]*2, y_pred_clone.shape[3]*2]), mode='bicubic')
    y_clone = F.interpolate(y_clone, size=tuple([y_clone.shape[2]*2, y_clone.shape[3]*2]), mode='bicubic')
    loss = (TVdist(y_pred_clone, y_clone, eps=eps) + TVdist(y_pred, y, eps=eps)) * .5
    return loss

def TVdist_list(y_pred, y, eps=0.00001):
    if type(y).__name__ == 'list':
        loss = 0
        for yy in y:
            loss = loss + TVdist(y_pred, yy)
        loss /= len(y)
    else:
        loss = TVdist(y_pred, y)
    return loss

def TVdist_list1(y_pred, y, eps=0.00001):
    if type(y).__name__ == 'list':
        loss1 = TVdist(y_pred, y[0])
        loss2 = -calc_nss_score(y_pred, y[1], eps=eps)
        loss = loss1 + loss2
    else:
        loss = TVdist(y_pred, y)
    return loss

def calc_nss_score(y_pred, y, eps=0.00001):    
    salMap = y_pred - torch.mean(y_pred, dim=[2,3], keepdim=True).expand_as(y_pred)
    sal_std = torch.std(salMap, dim=[2,3], keepdim=True).expand_as(salMap)
    salMap = salMap / (sal_std+eps)
    y_sum = torch.sum(y, dim=[1,2,3])
    nss = torch.sum(salMap*y, dim=[1,2,3])/(torch.sum(y, dim=[1,2,3])+eps)
    nss = torch.sum(nss)
    return nss

def Logitdist(y_pred, y, eps=0.00001):
    if (y_pred.shape[2] != y.shape[2] and 
        y_pred.shape[3] != y.shape[3]):
        y_pred = F.interpolate(y_pred, 
            size=tuple(y.size()[2:]), mode=upsample_mode)
    P = y / (eps + y.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    Q = y_pred / (eps + y_pred.sum(dim=[1, 2, 3]).view(-1,1,1,1))
    tv = torch.sum( -P*torch.log(Q+eps) )
    return tv

def NSS(input,fixation):
    input = input.view(input.size(0), -1)
    input = torch.div(input,input.max(1,keepdim=True)[0].expand_as(input) + epsilon)
    fixation = fixation.view(fixation.size(0),-1)
    input = torch.div(input-input.mean(1,keepdim=True).expand_as(input),input.std(1,keepdim=True).expand_as(input) + epsilon)
    loss = torch.div(torch.mul(input,fixation).sum(1), fixation.sum(1) + epsilon)
    return -torch.mean(loss)

def CC(input,fixmap): 
    input = input.view(input.size(0), -1)
    input = torch.div(input,input.max(1,keepdim=True)[0].expand_as(input) + epsilon)
    fixmap = fixmap.view(fixmap.size(0),-1)
    fixmap = torch.div(fixmap,fixmap.sum(1,keepdim=True).expand_as(fixmap) + epsilon)
    input = torch.div(input,input.sum(1,keepdim=True).expand_as(input) + epsilon)
    cov = torch.mul(input-input.mean(1,keepdim=True).expand_as(input),fixmap-fixmap.mean(1,keepdim=True).expand_as(fixmap)).mean(1)
    loss = torch.div(cov,torch.mul(input.std(1),fixmap.std(1)) + epsilon)
    return -torch.mean(loss)

def compositionalLoss(input,fixmap):
	loss = spatial_cross_entropy(input, fixmap)+CC(input,fixmap)
	return loss