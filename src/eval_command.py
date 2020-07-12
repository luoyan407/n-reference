import os, sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src')
sys.path.append(srcFolder)

from metrics import nss
from metrics import auc
from metrics import cc
from utils import *

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Evaluate predicted saliency map')
parser.add_argument('--output', type=str, default='')
parser.add_argument('--fixation_folder', type=str, default='')
parser.add_argument('--salmap_folder', type=str, default='')
parser.add_argument('--split_file', type=str, default='')
parser.add_argument('--fxt_loc_name', type=str, default='fixationPts')
parser.add_argument('--fxt_size', type=str, default='',
                    help='fixation resolution: (600, 800) | (480, 640) | (320, 640)')
parser.add_argument('--appendix', type=str, default='')
parser.add_argument('--file_extension', type=str, default='jpg')
args = parser.parse_args()

if args.fxt_size != '':
    spl_tokens = args.fxt_size.split()
    args.fxt_size = (int(spl_tokens[0]), int(spl_tokens[1]))
else:
    args.fxt_size = (480, 640)
fixation_folder = args.fixation_folder
salmap_folder = args.salmap_folder

fxtimg_type = detect_images_type(fixation_folder)
split_file = args.split_file
if split_file != '' and os.path.isfile(split_file):
    npzfile = np.load(split_file)
    salmap_names = [os.path.join(salmap_folder, x) for x in npzfile['val_imgs']]
    gtsal_names = [os.path.join(fixation_folder, x[:x.find('.')+1]+fxtimg_type) for x in npzfile['val_imgs']]
    fxtpts_names = [os.path.join(fixation_folder, '{}mat'.format(x[:x.find('.')+1])) for x in npzfile['val_imgs']]
else:
    salmap_names = load_allimages_list(salmap_folder)
    gtsal_names = []
    fxtpts_names = []
    for sn in salmap_names:
        file_name = sn.split('/')[-1]
        gtsal_names.append(os.path.join(fixation_folder,'{}{}'.format(file_name[:file_name.find('.')+1], fxtimg_type)))
        fxtpts_names.append(os.path.join(fixation_folder,'{}mat'.format(file_name[:file_name.find('.')+1])))

nss_score, _ = nss.compute_score(salmap_names, fxtpts_names, image_size=args.fxt_size, fxt_field_in_mat=args.fxt_loc_name)
cc_score, _ = cc.compute_score(salmap_names, gtsal_names, image_size=args.fxt_size)
auc_score, _ = auc.compute_score(salmap_names, fxtpts_names, image_size=args.fxt_size, fxt_field_in_mat=args.fxt_loc_name)

with open(args.output, 'a') as f:
    f.write('{:0.4f}, {:0.4f}, {:0.4f}{}\n'.format(
            nss_score, auc_score, cc_score, args.appendix))