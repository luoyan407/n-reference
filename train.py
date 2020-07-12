import torch
import pickle
import os.path
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import argparse
import subprocess

import collections
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
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of epochs')
parser.add_argument('--batch_size', default=10, type=int,
                    help='batch size for training')
parser.add_argument('--val_batch_size', default=1, type=int,
                    help='batch size for validation')
parser.add_argument('--model', default='resnet50', type=str,
                    help='backbone network: resnet50, din50')
parser.add_argument('--pretrainedModel', default='', type=str,
                    help='pretrained saliency model')
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
                    help='resized image resolution for training: (600, 800) | (480, 640) | (320, 640)')
parser.add_argument('--tr_fxt_size', nargs='+', type=int,
                    help='resized training fixation resolution: (600, 800) | (480, 640) | (320, 640)')
parser.add_argument('--val_fxt_size', nargs='+', type=int,
                    help='resized validation fixation resolution: (600, 800) | (480, 640) | (320, 640)')
parser.add_argument('--out_dir', default='logs/baseline/salicon_dinet', 
                    type=str,
                    help='validation saliency maps path')
parser.add_argument('--fxt_loc_name', type=str, default='fixationPts', help='fixationPts|fixLocs')
parser.add_argument('--random_seed', default=0, type=int,
                    help='random seed')
args = parser.parse_args()

args.start_epoch = 0
args.pretrained = True
args.useMultiGPU = True
n_output = 256 # just for DINet

args.experiment_name = '{}'.format(args.model)

out_folder = args.out_dir
args.loggerdir = '{}/tsboard'.format(out_folder)
args.save_path = '{}/snapshots'.format(out_folder)
args.sal_path = '{}/salmap_val'.format(out_folder)
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

# create the model
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

# As the output of saliency prediction is a saliency map, 
# whose size depends on the input size, we do a test here
# to quickly acquire the output size.
testImgs = load_allimages(args.val_img_dir)
oneimage = testImgs[0][0]
oneimage = datasets.folder.default_loader(oneimage)
oneimage = transforms.Resize(args.image_size)(oneimage)
oneimage = transforms.ToTensor()(oneimage)
oneimage = oneimage.view([1]+list(oneimage.size()))
oneimage = Variable(oneimage).cuda()
output = model(oneimage)

train_loader, val_loader = create_data_loaders(args,
                                            outSize=tuple(output.size()[2:]), 
                                            imgSize=args.image_size, 
                                            trFxtSize=args.tr_fxt_size,
                                            valFxtSize=args.val_fxt_size,
                                            flip=False)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

number_of_params = sum(p.numel() for p in model.parameters())
print('===Total parameters number: {}'.format(number_of_params))

snapshot_dir = args.save_path
ensure_dir(snapshot_dir)
pickle.dump(vars(args), open(snapshot_dir + 'args.pkl', 'wb'))

txtlogger = open('{}/log.txt'.format(out_folder), 'w')
print(vars(args),file=txtlogger, flush=True)
print(criterion,file=txtlogger, flush=True)

stat_file = os.path.join(out_folder, 'stat_training.csv')
with open(stat_file, 'w') as f:
    f.write('nss, auc, cc, ep, tr_loss, val_loss, tr_batchtime, tr_datatime, val_batchtime, val_datatime\n')

for epoch in range(args.start_epoch, args.epochs):
    adjust_learning_rate(args.lr, optimizer, epoch, 
       basenum=args.lr_decay_epoch, coef=args.lr_coef)
    cur_lr = optimizer.param_groups[0]['lr']

    train_loss, val_loss, train_batchtime, train_datatime, val_batchtime, val_datatime = train_val(model, criterion, optimizer, epoch,
                                    train_loader, val_loader, args.sal_path, txtlogger)
    
    save_filename = os.path.join(snapshot_dir, 'model_ep{epoch}.pth'.format(epoch=epoch+1))
    save_gem_filename = os.path.join(snapshot_dir, 'model_gem_ep{epoch}.pth'.format(epoch=epoch+1))
    save_checkpoint(model, save_filename)

    sal_path = '{}/ep{}'.format(args.sal_path, epoch+1)
    isnotify = 0 if epoch < args.epochs-1 else 1
    appendix = ', {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(epoch+1, train_loss, val_loss, train_batchtime, train_datatime, val_batchtime, val_datatime)
    evalCmd = 'python src/eval_command.py --output "{}" --fixation_folder "{}" --salmap_folder "{}" --fxt_loc_name "{}" --appendix "{}"'.format(stat_file, args.val_gt_dir, sal_path, args.fxt_loc_name, appendix)
    subprocess.Popen(evalCmd, shell=True)

txtlogger.close()
