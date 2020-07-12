# to view performance, tensorboard --logdir='./logs/resnet50/tsboard' --port=9000
# to run, CUDA_VISIBLE_DEVICES=0,1 python train_gsm.py
import torch
import pickle
import os.path
import torchvision
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import numpy as np
import argparse
import subprocess

import collections
from collections import OrderedDict
args = collections.namedtuple
import sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src')
sys.path.append(srcFolder)
# import gem
# import regressor_din
from metrics import (nss, auc, cc)
from utils import *
from losses import *
from models import *
from training_scheme import *

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Saliency Training')
parser.add_argument('--lr', default=0.0005, type=float,
                    help='learning rate')
parser.add_argument('--lr_decay_epoch', default=2, type=float,
                    help='every n epochs to decay learning rate')
parser.add_argument('--lr_coef', default=.1, type=float,
                    help='lr coefficient to change learning rates')
parser.add_argument('--weight_decay', default=0.0, type=float,
                    help='weight decay')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='momentum')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of epochs')
parser.add_argument('--batch_size', default=8, type=int,
                    help='batch size for training')
parser.add_argument('--val_batch_size', default=1, type=int,
                    help='batch size for validation')
parser.add_argument('--model', default='resnet50', type=str,
                    help='backbone network: resnet50, din50')

parser.add_argument('--pretrainedModel', default='', type=str,
                    help='pretrained saliency model')
parser.add_argument('--pretrainedModel_head', default='', type=str,
                    help='pretrained saliency model head')
parser.add_argument('--eval_mode', default='synchronous', type=str,
                    help='eval mode: synchronous|asynchronous')
parser.add_argument('--train_img_dir', nargs='+', default=[], 
                    help='training images path')
parser.add_argument('--train_gt_dir', nargs='+', default=[],
                    help='training human fixation maps path')
parser.add_argument('--val_img_dir', nargs='+', default=[],
                    help='validation images path')
parser.add_argument('--val_gt_dir', nargs='+', default=[],
                    help='validation human fixation maps path')
parser.add_argument('--image_size', nargs='+', type=int,
                    help='resized image resolution for training: (600, 800) | (480, 640) | (320, 640)')
parser.add_argument('--tr_fxt_size', nargs='+', type=int,
                    help='resized training fixation resolution: (600, 800) | (480, 640) | (320, 640)')
parser.add_argument('--val_fxt_size', nargs='+', type=int,
                    help='resized validation fixation resolution: (600, 800) | (480, 640) | (320, 640)')
parser.add_argument('--out_dir', default='logs/salicon_dinet_websal', 
                    type=str,
                    help='validation saliency maps path')
parser.add_argument('--num_shots', default=5, type=int,
                    help='number of shots')
parser.add_argument('--split_file', default='', 
                    type=str,
                    help='split file for n-shot learning, i.e., n samples are taken for training samples while the rest are taken for validating')
parser.add_argument('--fxt_loc_name', type=str, default='fixationPts', help='fixationPts|fixLocs')
parser.add_argument('--ref_split_layer', default=1, type=int,
                    help='number of the split layers')
parser.add_argument('--random_seed', default=0, type=int,
                    help='random seed')
args = parser.parse_args()

args.start_epoch = 0
pretrained = True
useMultiGPU = True
n_output = 256 # just for DINet

args.experiment_name = '{}'.format(args.model)

out_folder = args.out_dir
ensure_dir(out_folder)
args.save_path = '{}/snapshots'.format(out_folder)
args.sal_path = '{}/salmap_val'.format(out_folder)
split_file = '{}/split_data.npz'.format(out_folder) if args.split_file == '' else args.split_file
if args.image_size is None:
    args.image_size = (480,640)
else:
    args.image_size = (args.image_size[0], args.image_size[1])

modelzoo = {
    'densenet169': models.densenet169,
    'vgg16': models.vgg16,
    'resnet101': models.resnet101,
    'resnet50': models.resnet50,
    'resnet34': models.resnet34,
    'resnet18': models.resnet18,
}

print(vars(args))

criterion = TVdist

# create the model and optimizer
if args.model == 'din50':
    model = Saliency_DIN(args.model,modelzoo,pretrained,n_output=n_output)
elif args.model == 'resnet50':
    model = Saliency_ResNet50(args.model,modelzoo,pretrained)

if args.pretrainedModel != '' and args.pretrainedModel_head == '':
    model.load_state_dict(torch.load(args.pretrainedModel))
elif args.pretrainedModel != '' and args.pretrainedModel_head != '':
    model_body, model_head = split_model_din(model, split_layer=args.ref_split_layer)
    model_head = Referencer(model_head)
    model_body.load_state_dict(torch.load(args.pretrainedModel))
    model_head.load_state_dict(torch.load(args.pretrainedModel_head))
    layers_body = list(model_body.children())
    layers_head = list(model_head.children())
    model = nn.Sequential(*layers_body, *layers_head)

if not useMultiGPU:
    model = model.cuda()
elif useMultiGPU:
    model = nn.DataParallel(model).cuda()

# As the output of saliency prediction is a saliency map, 
# whose size depends on the input size, we do a test here
# to quickly acquire the output size.
testImgs = load_allimages(args.val_img_dir[0])
oneimage = testImgs[0][0]
oneimage = datasets.folder.default_loader(oneimage)
oneimage = transforms.Resize(args.image_size)(oneimage)
oneimage = transforms.ToTensor()(oneimage)
oneimage = oneimage.view([1]+list(oneimage.size()))
oneimage = Variable(oneimage).cuda()
output = model(oneimage)

# Due to n-shot learning setting, we split the validation set into the n samples for training
# and the rest of samples are for testing. So trFxtSize is the same as valFxtSize.
train_loader, val_loader, print_str = create_nshotsplit_loaders(args,
                            outSize=tuple(output.size()[2:]), 
                            imgSize=args.image_size, 
                            trFxtSize=args.val_fxt_size,
                            valFxtSize=args.val_fxt_size,
                            split_file=split_file,
                            num_shots=args.num_shots,
                            flip=False)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

#print total number of parameters
number_of_params = sum(p.numel() for p in model.parameters())

snapshot_dir = args.save_path
ensure_dir(snapshot_dir)
pickle.dump(vars(args), open(snapshot_dir + 'args.pkl', 'wb'))

txtlogger = open('{}/log.txt'.format(out_folder), 'w')
print(vars(args),file=txtlogger, flush=True)
print(criterion,file=txtlogger, flush=True)

print(print_str)
print(print_str,file=txtlogger, flush=True)

print('===Total parameters number: {}'.format(number_of_params))
print('===Total parameters number: {}'.format(number_of_params), file=txtlogger, flush=True)

stat_file = os.path.join(out_folder, 'stat_training.csv')
with open(stat_file, 'w') as f:
    f.write('nss, auc, cc, ep, tr_loss, val_loss, tr_batchtime, tr_datatime, val_batchtime, val_datatime\n')
if args.model in ['resnext50_din_cond', 'resnext50_din_sep', 'resnext50_din_sep1']:
    stat_cond_file = os.path.join(out_folder, 'stat_rec.csv')
    with open(stat_cond_file, 'w') as f:
        f.write('nss, auc, cc, ep, tr_loss, val_loss, tr_batchtime, tr_datatime, val_batchtime, val_datatime\n')

train_loss_list, val_loss_list, train_batchtime_list, train_datatime_list, val_batchtime_list, val_datatime_list = [], [], [], [], [], []
for epoch in range(args.start_epoch, args.epochs):
    adjust_learning_rate(args.lr, optimizer, epoch, 
       basenum=args.lr_decay_epoch, coef=args.lr_coef)
    cur_lr = optimizer.param_groups[0]['lr']

    sal_paths = ['{}/ep{}'.format(args.sal_path, epoch+1)]
    train_loss, val_loss, train_batchtime, train_datatime, val_batchtime, val_datatime = train_val_nshot(
                                    model, criterion, optimizer, epoch,
                                    train_loader, val_loader, sal_paths, txtlogger)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    train_batchtime_list.append(train_batchtime)
    train_datatime_list.append(train_datatime)
    val_batchtime_list.append(val_batchtime)
    val_datatime_list.append(val_datatime)
    
    save_filename = os.path.join(snapshot_dir, 'model_ep{epoch}.pth'.format(epoch=epoch+1))
    save_checkpoint(model, save_filename)

    if args.eval_mode == 'synchronous':
        sal_path = sal_paths[0]
        isnotify = 0 if epoch < args.epochs-1 else 1
        appendix = ', {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(epoch+1, train_loss, val_loss, train_batchtime, train_datatime, val_batchtime, val_datatime)
        evalCmd = 'python src/eval_command.py --output "{}" --fixation_folder "{}" --fxt_size "{} {}" --fxt_loc_name "{}" --salmap_folder "{}" --appendix "{}" --split_file "{}"'.format(stat_file, args.val_gt_dir[0], args.val_fxt_size[0], args.val_fxt_size[1], args.fxt_loc_name, sal_path, appendix, split_file)
        sproc = subprocess.Popen(evalCmd, shell=True)

txtlogger.close()

if args.eval_mode == 'asynchronous':
    for epoch in range(args.start_epoch, args.epochs):
        sal_path = '{}/ep{}'.format(args.sal_path, epoch+1)
        isnotify = 0 if epoch < args.epochs-1 else 1
        appendix = ', {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
            epoch+1, train_loss_list, val_loss_list, train_batchtime_list, train_datatime_list, val_batchtime_list, val_datatime_list)
        evalCmd = 'python src/eval_command.py --output "{}" --fixation_folder "{}" --salmap_folder "{}" --appendix "{}"'.format(
            stat_file, args.val_gt_dir, sal_path, appendix)
        sproc = subprocess.Popen(evalCmd, shell=True)
        sproc.wait()