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

parser = argparse.ArgumentParser(description='Blur fixations')
parser.add_argument('--input', type=str, default='')
parser.add_argument('--output', type=str, default='')
parser.add_argument('--sigma', type=float, default=19.0)
parser.add_argument('--type', type=str, default='png')

args = parser.parse_args()

input_folder = args.input
output_folder = args.output
sigma = args.sigma
image_type = args.type

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

fxtpts_names = load_allimages_list_norec(input_folder)

ncount = 0
for i in range(len(fxtpts_names)):
    fname = fxtpts_names[i]
    ncount += 1
    if ncount % 1000 == 0:
        print('process the {}-th sample'.format(ncount))
    tokens = fname.split('/')
    fxtmap_name = os.path.join(output_folder, tokens[-1][:-3]+image_type)
    mat_name = os.path.join(output_folder, tokens[-1][:-3]+'mat')
    
    im = np.array(Image.open(fname))
    im = im/255
    fixationPts = im.astype(np.uint8)
    scipy.io.savemat(mat_name, {'fixationPts':fixationPts})

    fixations = fixationPts.astype(float)
    fixations = scipy.ndimage.filters.gaussian_filter(fixations, sigma)
    fixations -= np.min(fixations)
    if np.max(fixations) > 0:
        fixations = fixations / np.max(fixations)*255.0
    cv2.imwrite(fxtmap_name,fixations)
