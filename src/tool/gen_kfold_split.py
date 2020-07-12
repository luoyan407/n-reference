import os, sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../src')
sys.path.append(srcFolder)

from utils import *

import numpy as np
import argparse
import scipy.io
import scipy.misc
import scipy.ndimage
import scipy.ndimage.filters
from PIL import Image
import shutil 
import cv2
import random

parser = argparse.ArgumentParser(description='Split data generation')
parser.add_argument('--image', type=str, default='')
parser.add_argument('--output', type=str, default='')
parser.add_argument('--k', type=int, default=3)

args = parser.parse_args()

input_folder = args.image
output_folder = args.output
k = args.k

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_names = load_allimages_wopath(input_folder)
random.shuffle(image_names)
num_subset = int(len(image_names)/k)

subset_list = [image_names[:num_subset], image_names[num_subset:2*num_subset], image_names[2*num_subset:]]

for i in range(args.k):

    valset = subset_list[i]
    trainset = []
    for j in range(args.k):
        if j != i:
            trainset += subset_list[j]
    split_file = os.path.join(output_folder, 'split_s{}.npz'.format(i+1))
    print(len(trainset), len(valset))
    np.savez(split_file, train_imgs=np.array(trainset), val_imgs=np.array(valset))
