import torch
import pickle
import os.path
import torchvision
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import argparse
import subprocess

import collections
from collections import OrderedDict
args = collections.namedtuple
import sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src')
sys.path.append(srcFolder)
from metrics import (nss, auc, cc)
from utils import *
from losses import *
from models import *
from training_scheme import *

parser = argparse.ArgumentParser(description='Saliency Training')
parser.add_argument('--lr', default=0.0005, type=float,
                    help='learning rate')
parser.add_argument('--lr_decay_epoch', default=2, type=float,
                    help='every n epochs to decay learning rate')
parser.add_argument('--lr_coef', default=.1, type=float,
                    help='lr coefficient to change learning rates')
parser.add_argument('--weight_decay', default=0.0, type=float,
                    help='weight decay')
parser.add_argument('--epoch', default=10, type=int,
                    help='number of epochs')
parser.add_argument('--batchsize', default=8, type=int,
                    help='batch size for training')
parser.add_argument('--val_batchsize', default=1, type=int,
                    help='batch size for validation')
parser.add_argument('--model', default='resnet50', type=str,
                    help='backbone network: resnet50, din50')

parser.add_argument('--type_dispatcher', default=0, type=int,
                    help='0: use baseline, 1: l2 norm, 2: angle')
parser.add_argument('--memsize', default=5, type=int,
                    help='memory size')
parser.add_argument('--mem_window', default=0, type=int,
                    help='reset window')
parser.add_argument('--mem_offset', default=0, type=int,
                    help='memory offset')
parser.add_argument('--mem_strength', default=0.0, type=float,
                    help='memory strength')
parser.add_argument('--mtype', default='dcl', type=str,
                    help='method type: baseline, dcl, gem')

parser.add_argument('--train_img_dir', default='', 
                    type=str,
                    help='training images path')
parser.add_argument('--train_gt_dir', default='', 
                    type=str,
                    help='training human fixation maps path')
parser.add_argument('--val_img_dir', default='', 
                    type=str,
                    help='validation images path')
parser.add_argument('--val_gt_dir', default='', 
                    type=str,
                    help='validation human fixation maps path')
parser.add_argument('--image_size', nargs='+', type=int,
                    help='image resolution for training: (600, 800) | (480, 640) | (320, 640)')
parser.add_argument('--out_dir', default='', 
                    type=str,
                    help='validation human fixation maps path')
parser.add_argument('--optype', default='adam', 
                    type=str,
                    help='optimizer')
parser.add_argument('--random_seed', default=0, type=int,
                    help='random seed')
parser.add_argument('--pretrainedModel', default='', 
                    type=str,
                    help='pretrained saliency model path')
parser.add_argument('--gaussblur_sigma', default=-1.0, type=float,
                    help='Gaussian blur sigma')
parser.add_argument('--gaussblur_truncate', default=4.0, type=float,
                    help='Gaussian blur truncate')
parser.add_argument('--file_type', default='jpg', type=str,
                    help='output file type')

args = parser.parse_args()

args.start_epoch = 0


args.pretrained = True
args.useMultiGPU = True

args.pretrainedModel = ''

args.experiment_name = '{}'.format(args.model)

out_folder = args.out_dir
ensure_dir(out_folder)
if args.image_size is None:
    args.image_size = (480,640)
else:
    args.image_size = (args.image_size[0], args.image_size[1])

n_output = 256

modelzoo = {
    'densenet169': models.densenet169,
    'vgg16': models.vgg16,
    'resnet101': models.resnet101,
    'resnet50': models.resnet50,
    'resnet34': models.resnet34,
    'resnet18': models.resnet18,
}

print(vars(args))

# create the model and optimizer
if args.model == 'din50':
    model = Saliency_DIN(args.model,modelzoo,args.pretrained,n_output=n_output)
elif args.model == 'resnet50':
    model = Saliency_ResNet50(args.model,modelzoo,args.pretrained)

if args.pretrainedModel != '':
    model.load_state_dict(torch.load(args.pretrainedModel))

if not args.useMultiGPU:
    model = model.cuda()
    param_groups = model.get_param_groups()
elif args.useMultiGPU:
    model = nn.DataParallel(model).cuda()

val_loader = create_test_data_loader(args.val_img_dir, 
                                    args.val_batch_size,
                                    args.image_size)

#print total number of parameters
number_of_params = sum(p.numel() for p in model.parameters())
print('===Total parameters number: {}'.format(number_of_params))

txtlogger = open('{}/predict_stat.txt'.format(out_folder), 'w')
print(vars(args),file=txtlogger, flush=True)

val_batchtime, val_datatime = predict(model, val_loader, os.path.join(out_folder, 'salmaps'), 
                                inargs.gaussblur_sigma, inargs.gaussblur_truncate, inargs.file_type)
print('avg batch time {}, avg data time {}, avg proc time {}'.format(val_batchtime, val_datatime, val_batchtime-val_datatime),file=txtlogger, flush=True)
print('avg batch time {}, avg data time {}, avg proc time {}'.format(val_batchtime, val_datatime, val_batchtime-val_datatime))

txtlogger.close()
