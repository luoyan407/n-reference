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

import kornia

import numpy as np
import time
import random
import scipy
import cv2
import math
import collections
import scipy.io
import scipy.ndimage
import scipy.misc

from PIL import Image, ImageOps

import os, sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src')
sys.path.append(srcFolder)
from models import *
from utils import *
from losses import *

class SaliconDataset(data.Dataset):
    """Dataset wrapping images and saliency maps."""
    
    def __init__(self, img_path, map_path=None, 
        size=None, loader=datasets.folder.default_loader, flip=False, 
        outSize=(30,40), imgSize=(480,640), fxtSize=(480,640)):
        self.imgs = load_allimages(img_path)
        self.imgs.sort(key = lambda t: t[0])
        self.img_path = img_path
        self.img_transform = transforms.Compose([
            transforms.Resize(imgSize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.outSize = outSize
        self.imgSize = imgSize

        if fxtSize is None:
            fxtSize = (480,640)
        
        self.map_path = map_path
        if map_path:
            self.maps = load_allimages(map_path)
            self.maps.sort(key = lambda t: t[0])

            self.map_transform = transforms.Compose([
                transforms.Resize(fxtSize),
                transforms.ToTensor()
            ])
        
        self.loader = loader
        self.flip = flip
        
        import random
        if size:
            shuffled = random.sample(list(zip(self.imgs, self.maps)), size)
            self.imgs, self.maps = tuple(map(list, zip(*shuffled)))
            
    def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, fixmap) where map is the fixation map of the image.
            """

            rnd = random.random()
            
            pathnames = self.imgs[index][0]
            img = self.loader(self.imgs[index][0])
            if self.flip and rnd < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.img_transform(img)

            fixmap = torch.zeros(1)
            if self.map_path:

                fixmap = self.loader(self.maps[index][0])
                if self.flip and rnd < 0.5:
                    fixmap = fixmap.transpose(Image.FLIP_LEFT_RIGHT)
                fixmap = self.map_transform(fixmap)[0,:,:]

                fixmap /= fixmap.sum()
                fixmap = fixmap.unsqueeze(0)

            return img, fixmap, pathnames

    def __len__(self):
        return len(self.imgs)

class NShotDataset(data.Dataset):
    """Dataset wrapping images and saliency maps."""
    
    def __init__(self, img_path, map_path=None, images_list=None,
        size=None, loader=datasets.folder.default_loader, flip=False, 
        outSize=(30,40), imgSize=(480,640), fxtSize=(480,640),
        split=''):
        if images_list is None:
            self.imgs = load_allimages_list_norec(img_path)
            self.maps = load_allimages_list_norec(map_path)
        else:
            self.imgs = [os.path.join(img_path, x) for x in images_list]
            self.maps = [os.path.join(map_path, x) for x in images_list]

        self.imgs.sort(key = lambda t: t[0])
        self.img_path = img_path
        self.img_transform = transforms.Compose([
            transforms.Resize(imgSize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.outSize = outSize
        self.imgSize = imgSize
        
        self.map_path = map_path
        self.maps.sort(key = lambda t: t[0])

        self.map_transform = transforms.Compose([
            transforms.Resize(fxtSize),
            transforms.ToTensor()
        ])
        
        self.loader = loader
        self.flip = flip
        
        import random
        if size:
            shuffled = random.sample(list(zip(self.imgs, self.maps)), size)
            self.imgs, self.maps = tuple(map(list, zip(*shuffled)))
            
    def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, fixmap) where map is the fixation map of the image.
            """

            rnd = random.random()
            
            pathnames = self.imgs[index]
            img = self.loader(pathnames)
            if self.flip and rnd < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.img_transform(img)

            fixmap = torch.zeros(1)
            if self.map_path:

                fixmap = self.loader(self.maps[index])
                if self.flip and rnd < 0.5:
                    fixmap = fixmap.transpose(Image.FLIP_LEFT_RIGHT)
                fixmap = self.map_transform(fixmap)[0,:,:]

                fixmap /= fixmap.sum()
                fixmap = fixmap.unsqueeze(0)

            return img, fixmap, pathnames

    def __len__(self):
        return len(self.imgs)

class NShotDataset_with_path(data.Dataset):
    """Dataset wrapping images and saliency maps."""
    
    def __init__(self, images_list=None, fxts_list=None,
        size=None, loader=datasets.folder.default_loader, flip=False, 
        outSize=(30,40), imgSize=(480,640), fxtSize=(480,640),
        split=''):
        
        self.imgs = images_list
        self.maps = fxts_list

        self.imgs.sort(key = lambda t: t[0])
        self.img_transform = transforms.Compose([
            transforms.Resize(imgSize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.outSize = outSize
        self.imgSize = imgSize
        
        self.maps.sort(key = lambda t: t[0])

        self.map_transform = transforms.Compose([
            transforms.Resize(fxtSize),
            transforms.ToTensor()
        ])
        
        self.loader = loader
        self.flip = flip
        
        import random
        if size:
            shuffled = random.sample(list(zip(self.imgs, self.maps)), size)
            self.imgs, self.maps = tuple(map(list, zip(*shuffled)))
            
    def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, fixmap) where map is the fixation map of the image.
            """

            rnd = random.random()
            
            pathnames = self.imgs[index]
            img = self.loader(pathnames)
            if self.flip and rnd < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.img_transform(img)
               
            fixmap = self.loader(self.maps[index])
            if self.flip and rnd < 0.5:
                fixmap = fixmap.transpose(Image.FLIP_LEFT_RIGHT)
            fixmap = self.map_transform(fixmap)[0,:,:]
            fixmap /= fixmap.sum()
            fixmap = fixmap.unsqueeze(0)

            return img, fixmap, pathnames

    def __len__(self):
        return len(self.imgs)

def create_refdata_loaders(ref_img_dir,
                        ref_gt_dir,
                        ref_size=None, 
                        expected_size=None,
                        outSize=None, 
                        imgSize=None,
                        fxtSize=None,
                        shuffle=True,
                        flip=False,
                        numWorkers=4,
                        batch_size=1):
    shuffle_opt = shuffle
    data_source = SaliconDataset(ref_img_dir,
                       ref_gt_dir, 
                       size=ref_size, flip=flip,outSize=outSize,imgSize=imgSize,fxtSize=fxtSize)
    if ref_size is None:
        ref_size = len(data_source.imgs)
    custom_sampler = ExtendedRandomSampler(ref_size, expected_size)
    ref_loader = torch.utils.data.DataLoader(
        data_source,
        batch_size=batch_size,
        shuffle=False,
        sampler=custom_sampler,
        num_workers=numWorkers
    )

    print('*Stats* reference img number:{}, batch size:{}'.format(len(ref_loader.dataset.imgs), batch_size))
    return ref_loader

def create_external_loaders(ref_img_dir,
                        ref_gt_dir,
                        train_size=None, 
                        val_size=None, 
                        outSize=None, 
                        imgSize=None,
                        shuffle=True,
                        flip=True,
                        numWorkers=4):
    data_source = ExternalDataset_multi(ref_img_dir,
                       ref_gt_dir, 
                       size=train_size, flip=flip,outSize=outSize,imgSize=imgSize)
    ref_loader = torch.utils.data.DataLoader(
        data_source,
        batch_size=1,
        shuffle=False,
        num_workers=numWorkers
    )

    print('*Stats* external img number:{}, batch size:{}'.format(len(ref_loader.dataset.imgs), 1))
    return ref_loader

class SaliconDataset_multi(data.Dataset):
    """Dataset wrapping images and saliency maps."""
    
    def __init__(self, img_path, map_path=None, 
        size=None, loader=datasets.folder.default_loader, flip=False, 
        outSize=(30,40), imgSize=(480,640), fxtSize=(480,640)):
        tmp = []
        if type(img_path).__name__ == 'list':
            for p1 in img_path:
                tmp += load_allimages_list_norec(p1)
        else:
            tmp = load_allimages_list_norec(img_path)
        self.imgs = tmp
        self.imgs.sort(key = lambda t: t[0])
        self.img_path = img_path
        self.img_transform = transforms.Compose([
            transforms.Resize(imgSize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.outSize = outSize
        self.imgSize = imgSize

        if fxtSize is None:
            fxtSize = (480,640)
        
        self.map_path = map_path
        if map_path:
            tmp = []
            if type(map_path).__name__ == 'list':
                for p1 in map_path:
                    tmp += load_allimages_list_norec(p1)
            else:
                tmp = load_allimages_list_norec(map_path)
            self.maps = tmp
            self.maps.sort(key = lambda t: t[0])

            self.map_transform = transforms.Compose([
                transforms.Resize(fxtSize),
                transforms.ToTensor()
            ])
        
        self.loader = loader
        self.flip = flip
        
        import random
        if size:
            shuffled = random.sample(zip(self.imgs, self.maps), size)
            self.imgs, self.maps = tuple(map(list, zip(*shuffled)))
            


    def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, fixmap) where map is the fixation map of the image.
            """

            rnd = random.random()
            
            pathname = self.imgs[index]
            img = self.loader(pathname)
            if self.flip and rnd < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.img_transform(img)

            fixmap = torch.zeros(1)
            if self.map_path:
                map_pathname = self.maps[index]
                fixmap = self.loader(map_pathname)
                if self.flip and rnd < 0.5:
                    fixmap = fixmap.transpose(Image.FLIP_LEFT_RIGHT)
                fixmap = self.map_transform(fixmap)[0,:,:]

                fixmap /= fixmap.sum()
                fixmap = fixmap.unsqueeze(0)

            return img, fixmap, pathname

    def __len__(self):
        return len(self.imgs)

class SaliconDataset_sim(data.Dataset):
    """Dataset wrapping images and saliency maps."""
    
    def __init__(self, similarity_stat_file, similarity_threshold,
        size=None, loader=datasets.folder.default_loader, flip=False, 
        outSize=(30,40), imgSize=(480,640), fxtSize=(480,640)):
        npzfile = np.load(similarity_stat_file)
        ext_img_paths = npzfile['ext_img_paths']
        ext_fxt_paths = npzfile['ext_fxt_paths']
        ext_img_sims = npzfile['ext_img_sims']
        ext_img_grads = npzfile['ext_img_grads']
        reference_grad = npzfile['reference_grad']
        indices = ext_img_sims <= similarity_threshold
        
        ext_img_paths = ext_img_paths[indices]
        ext_fxt_paths = ext_fxt_paths[indices]
        ext_img_paths = [x[0] for x in ext_img_paths.tolist()]
        ext_fxt_paths = [x[0] for x in ext_fxt_paths.tolist()]

        self.imgs = ext_img_paths
        self.imgs.sort()
        self.img_transform = transforms.Compose([
            transforms.Resize(imgSize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.outSize = outSize
        self.imgSize = imgSize
        
        if ext_fxt_paths is not None:
            
            self.maps = ext_fxt_paths
            self.maps.sort()

            self.map_transform = transforms.Compose([
                transforms.Resize(fxtSize),
                transforms.ToTensor()
            ])
        
        self.loader = loader
        self.flip = flip
        
        import random
        if size:
            shuffled = random.sample(zip(self.imgs, self.maps), size)
            self.imgs, self.maps = tuple(map(list, zip(*shuffled)))
            
    def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, fixmap) where map is the fixation map of the image.
            """

            rnd = random.random()
            
            pathnames = self.imgs[index]
            img = self.loader(self.imgs[index])
            if self.flip and rnd < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.img_transform(img)

            fixmap = torch.zeros(1)
            if self.maps is not None:

                fixmap = self.loader(self.maps[index])
                if self.flip and rnd < 0.5:
                    fixmap = fixmap.transpose(Image.FLIP_LEFT_RIGHT)
                fixmap = self.map_transform(fixmap)[0,:,:]

                fixmap /= fixmap.sum()
                fixmap = fixmap.unsqueeze(0)

            return img, fixmap, pathnames

    def __len__(self):
        return len(self.imgs)

class ExternalDataset_multi(data.Dataset):
    """Dataset wrapping images and saliency maps."""
    
    def __init__(self, img_path, map_path=None, 
        size=None, loader=datasets.folder.default_loader, flip=False, 
        outSize=(30,40), imgSize=(480,640), fxtSize=(480,640)):
        tmp = []
        if type(img_path).__name__ == 'list':
            for p1 in img_path:
                tmp += load_allimages(p1)
        else:
            tmp = load_allimages(img_path)
        self.imgs = tmp
        self.imgs.sort(key = lambda t: t[0])
        self.img_path = img_path
        self.img_transform = transforms.Compose([
            transforms.Resize(imgSize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.outSize = outSize
        self.imgSize = imgSize
        
        self.map_path = map_path
        if map_path:
            tmp = []
            if type(map_path).__name__ == 'list':
                for p1 in map_path:
                    tmp += load_allimages(p1)
            else:
                tmp = load_allimages(map_path)
            self.maps = tmp
            self.maps.sort(key = lambda t: t[0])

            self.map_transform = transforms.Compose([
                transforms.Resize(fxtSize),
                transforms.ToTensor()
            ])
        
        self.loader = loader
        self.flip = flip
        
        import random
        if size:
            shuffled = random.sample(zip(self.imgs, self.maps), size)
            self.imgs, self.maps = tuple(map(list, zip(*shuffled)))
            
    def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, fixmap) where map is the fixation map of the image.
            """

            rnd = random.random()
            
            pathnames = [self.imgs[index][0], self.maps[index][0]]
            img = self.loader(self.imgs[index][0])
            if self.flip and rnd < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.img_transform(img)

            fixmap = torch.zeros(1)
            if self.map_path:

                fixmap = self.loader(self.maps[index][0])
                if self.flip and rnd < 0.5:
                    fixmap = fixmap.transpose(Image.FLIP_LEFT_RIGHT)
                fixmap = self.map_transform(fixmap)[0,:,:]

                fixmap /= fixmap.sum()
                fixmap = fixmap.unsqueeze(0)

            return img, fixmap, pathnames[0], pathnames[1]

    def __len__(self):
        return len(self.imgs)

class SaliconDataset_nGT(data.Dataset):
    """Dataset wrapping images without fixation maps."""
    
    def __init__(self, img_path, 
        size=None, loader=datasets.folder.default_loader, flip=False, 
        outSize=(30,40), imgSize=(480,640)):
        self.imgs = load_allimages_list_norec(img_path)
        
        self.imgs.sort(key = lambda t: t[0])
        self.img_path = img_path
        self.img_transform = transforms.Compose([
            transforms.Resize(imgSize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.outSize = outSize
        self.imgSize = imgSize    
    
        self.loader = loader
        self.flip = flip
        
        import random
    
    def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, fname) 
            """

            rnd = random.random()
            
            pathname = self.imgs[index]
            img = self.loader(pathname)
            if self.flip and rnd < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_size = (img.size[1], img.size[0]) # (height, width)
            img = self.img_transform(img)

            return img, pathname, img_size

    def __len__(self):
        return len(self.imgs)

def create_multidb_loaders(args, 
                        train_size=None, 
                        val_size=None, 
                        outSize=None, 
                        imgSize=None,
                        trFxtSize=None,
                        valFxtSize=None,
                        shuffle=True,
                        flip=True,
                        numWorkers=4):
    shuffle_opt = shuffle
    data_source = SaliconDataset_multi(args.train_img_dir,
                       args.train_gt_dir, 
                       size=train_size, flip=flip,
                       outSize=outSize,imgSize=imgSize,fxtSize=trFxtSize)
    custom_sampler = None
    train_loader = torch.utils.data.DataLoader(
        data_source,
        batch_size=args.batch_size,
        shuffle=shuffle_opt,
        sampler=custom_sampler,
        num_workers=numWorkers
    )

    val_loader = torch.utils.data.DataLoader(
        SaliconDataset_multi(args.val_img_dir,
                       args.val_gt_dir, 
                       size=val_size, outSize=outSize,
                       imgSize=imgSize, flip=False, fxtSize=valFxtSize),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=numWorkers
    )
    print('*Stats* training img number:{}, val img number:{}, batch size:{}'.format(len(train_loader.dataset.imgs), len(val_loader.dataset.imgs), args.batch_size))
    return (train_loader, val_loader)

def select_by_sim_loaders(args, 
                        train_size=None, 
                        val_size=None, 
                        outSize=None, 
                        imgSize=None,
                        shuffle=True,
                        flip=True,
                        numWorkers=4):
    shuffle_opt = shuffle
    data_source = SaliconDataset_sim(args.similarity_stat_file,
                       args.similarity_threshold, 
                       size=train_size, flip=flip,outSize=outSize,imgSize=imgSize)
    custom_sampler = None
    train_loader = torch.utils.data.DataLoader(
        data_source,
        batch_size=args.batch_size,
        shuffle=shuffle_opt,
        sampler=custom_sampler,
        num_workers=numWorkers
    )

    val_loader = torch.utils.data.DataLoader(
        SaliconDataset_multi(args.val_img_dir,
                       args.val_gt_dir, 
                       size=val_size, outSize=outSize,imgSize=imgSize, flip=False),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=numWorkers
    )
    print('*Stats* training img number:{}, val img number:{}, batch size:{}'.format(len(train_loader.dataset.imgs), len(val_loader.dataset.imgs), args.batch_size))
    return (train_loader, val_loader)

def select_by_sim_loader(args, 
                        train_size=None, 
                        val_size=None, 
                        outSize=None, 
                        imgSize=None,
                        shuffle=True,
                        flip=True,
                        numWorkers=4):
    shuffle_opt = shuffle
    data_source = SaliconDataset_sim(args.similarity_stat_file,
                       args.similarity_threshold, 
                       size=train_size, flip=flip,outSize=outSize,imgSize=imgSize)
    custom_sampler = None
    train_loader = torch.utils.data.DataLoader(
        data_source,
        batch_size=args.batch_size,
        shuffle=shuffle_opt,
        sampler=custom_sampler,
        num_workers=numWorkers
    )

    print('*Stats* training img number:{}, batch size:{}'.format(len(train_loader.dataset.imgs), args.batch_size))
    return train_loader

def create_data_loaders(args, 
                        train_size=None, 
                        val_size=None, 
                        outSize=None, 
                        imgSize=None,
                        trFxtSize=None,
                        valFxtSize=None,
                        shuffle=True,
                        flip=True,
                        numWorkers=4):
    shuffle_opt = shuffle
    data_source = SaliconDataset('%s' % args.train_img_dir,
                       '%s' % args.train_gt_dir, 
                       size=train_size, flip=flip, outSize=outSize, imgSize=imgSize, fxtSize=trFxtSize)
    custom_sampler = None
    train_loader = torch.utils.data.DataLoader(
        data_source,
        batch_size=args.batch_size,
        shuffle=shuffle_opt,
        sampler=custom_sampler,
        num_workers=numWorkers
    )

    val_loader = torch.utils.data.DataLoader(
        SaliconDataset('%s' % args.val_img_dir,
                       '%s' % args.val_gt_dir, 
                       size=val_size, outSize=outSize, imgSize=imgSize, flip=False, fxtSize=valFxtSize),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=numWorkers
    )
    print('*Stats* training img number:{}, val img number:{}, batch size:{}'.format(len(train_loader.dataset.imgs), len(val_loader.dataset.imgs), args.batch_size))
    return (train_loader, val_loader)

def create_nshot_cat2000_loaders(args, 
                        num_shots=0,
                        split_file='',
                        train_size=None, 
                        val_size=None, 
                        outSize=None, 
                        imgSize=None,
                        trFxtSize=None,
                        valFxtSize=None,
                        subset_idx=0,
                        split_mode='tr',
                        ref_batch_size=1,
                        shuffle=True,
                        flip=True,
                        numWorkers=4):
    shuffle_opt = shuffle
    subsets = next(os.walk(args.train_img_dir[0]))[1]
    subsets.sort(key = lambda t: t[0])
    
    if os.path.isfile(split_file):
        npzfile = np.load(split_file)
        train_imgs = npzfile['train_imgs'].tolist()
        train_gts = npzfile['train_gts'].tolist()
        val_imgs = npzfile['val_imgs'].tolist()
        val_gts = npzfile['val_gts'].tolist()
        ref_imgs = npzfile['ref_imgs'].tolist()
        ref_gts = npzfile['ref_gts'].tolist()
    else:
        train_imgs = []
        train_gts = []
        for i,s in enumerate(subsets):
            if i != subset_idx:
                train_imgs += [os.path.join(args.train_img_dir[0],s,x) for x in load_allimages_wopath(os.path.join(args.train_img_dir[0],s))]
                train_gts += [os.path.join(args.train_gt_dir[0],s,x) for x in load_allimages_wopath(os.path.join(args.train_gt_dir[0],s))]
        val_imgs = [os.path.join(args.val_img_dir[0],subsets[subset_idx],x) for x in load_allimages_wopath(os.path.join(args.val_img_dir[0],subsets[subset_idx]))]
        val_gts = [os.path.join(args.val_gt_dir[0],subsets[subset_idx],x) for x in load_allimages_wopath(os.path.join(args.val_gt_dir[0],subsets[subset_idx]))]
        rd_indices = np.arange(len(val_imgs))
        np.random.shuffle(rd_indices)
        val_imgs = np.array(val_imgs)   
        val_gts = np.array(val_gts)
        ref_imgs, val_imgs = val_imgs[rd_indices[:num_shots]].tolist(), val_imgs[rd_indices[num_shots:]].tolist()
        ref_gts, val_gts = val_gts[rd_indices[:num_shots]].tolist(), val_gts[rd_indices[num_shots:]].tolist()

        np.savez(split_file, train_imgs=np.array(train_imgs), val_imgs=np.array(val_imgs), ref_imgs=np.array(ref_imgs),
                            train_gts=np.array(train_gts), val_gts=np.array(val_gts), ref_gts=np.array(ref_gts))
    if split_mode == 'tr':
        val_imgs += ref_imgs
        val_gts += ref_gts
        ref_imgs = None
        ref_gts = None
    elif split_mode == 'ft':
        train_imgs = ref_imgs
        train_gts = ref_gts
        ref_imgs = None
        ref_gts = None

    custom_sampler = None
    data_source = NShotDataset_with_path(images_list=train_imgs,
                   fxts_list=train_gts, 
                   size=val_size, outSize=outSize,
                   imgSize=imgSize, flip=False, fxtSize=trFxtSize)
    train_loader = torch.utils.data.DataLoader(
        data_source,
        batch_size=args.batch_size,
        shuffle=shuffle_opt,
        sampler=custom_sampler,
        num_workers=numWorkers
    )

    val_loader = torch.utils.data.DataLoader(
        NShotDataset_with_path(images_list=val_imgs,
                        fxts_list=val_imgs,
                       size=val_size, outSize=outSize,
                       imgSize=imgSize, flip=False, fxtSize=valFxtSize),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=numWorkers
    )

    ref_loader = None
    ref_og_size = 0
    print_str = ''
    if ref_imgs is not None:
        if len(ref_imgs) <= 10:
            print_str += '-----training images------\n'
            print_str += ','.join(ref_imgs)
            print_str += '\n--------------------------\n'
        ref_imgs = np.array(ref_imgs)
        ref_gts = np.array(ref_gts)
        ref_og_size = ref_imgs.shape[0]
        num_batches = int(np.ceil(data_source.__len__()/args.batch_size))
        sample_ints = np.random.randint(ref_imgs.shape[0], size=num_batches*ref_batch_size)
        ref_imgs = ref_imgs[sample_ints]
        ref_gts = ref_gts[sample_ints]
        ref_loader = torch.utils.data.DataLoader(
            NShotDataset_with_path(images_list=ref_imgs.tolist(),
                            fxts_list=ref_gts.tolist(), 
                           size=val_size, outSize=outSize,
                           imgSize=imgSize, flip=False, fxtSize=valFxtSize),
            batch_size=ref_batch_size,
            shuffle=False,
            num_workers=numWorkers
        )
    print_str += '*Stats* training img number:{}, val img number:{}, ref img number:{}, batch size:{}, val set:{}'.format(len(train_loader.dataset.imgs), len(val_loader.dataset.imgs), ref_og_size, args.batch_size, subsets[subset_idx])
    return (train_loader, val_loader, ref_loader, print_str, subsets[subset_idx])

def create_nshot_loaders(args, 
                        num_shots=5,
                        split_file='',
                        train_size=None, 
                        val_size=None, 
                        outSize=None, 
                        imgSize=None,
                        trFxtSize=None,
                        valFxtSize=None,
                        ref_batch_size=1,
                        shuffle=True,
                        flip=True,
                        numWorkers=4,
                        swap_ref_data=False):
    shuffle_opt = shuffle
    if num_shots < ref_batch_size:
        ref_batch_size = num_shots
    data_source = SaliconDataset_multi(args.train_img_dir,
                       args.train_gt_dir, 
                       size=train_size, flip=flip,
                       outSize=outSize,imgSize=imgSize,fxtSize=trFxtSize)
    custom_sampler = None
    train_loader = torch.utils.data.DataLoader(
        data_source,
        batch_size=args.batch_size,
        shuffle=shuffle_opt,
        sampler=custom_sampler,
        num_workers=numWorkers
    )

    if os.path.isfile(split_file):
        npzfile = np.load(split_file)
        train_imgs = npzfile['train_imgs']
        val_imgs = npzfile['val_imgs']
        train_imgs = [x for x in train_imgs.tolist()]
        val_imgs = [x for x in val_imgs.tolist()]
    else:
        image_names = load_allimages_wopath(args.val_img_dir[0])
        random.shuffle(image_names)
        train_imgs, val_imgs = image_names[:num_shots], image_names[num_shots:]

        np.savez(split_file, train_imgs=np.array(train_imgs), val_imgs=np.array(val_imgs))

    if swap_ref_data:
        tmp = train_imgs
        train_imgs = val_imgs
        val_imgs = tmp

    val_loader = torch.utils.data.DataLoader(
        NShotDataset(args.val_img_dir[0],
                       args.val_gt_dir[0], 
                       images_list=val_imgs,
                       size=val_size, outSize=outSize,
                       imgSize=imgSize, flip=False, fxtSize=valFxtSize),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=numWorkers
    )
    print_str = ''
    if len(train_imgs) <= 10:
        print_str += '-----training images------\n'
        print_str += ','.join(train_imgs)
        print_str += '\n--------------------------\n'
    train_imgs = np.array(train_imgs)
    ref_og_size = train_imgs.shape[0]
    num_batches = int(np.ceil(data_source.__len__()/args.batch_size))
    sample_ints = [np.random.randint(train_imgs.shape[0], size=ref_batch_size) for i in range(num_batches)]
    sample_ints = np.concatenate(sample_ints)
    train_imgs = train_imgs[sample_ints]
    ref_loader = torch.utils.data.DataLoader(
        NShotDataset(args.val_img_dir[0],
                       args.val_gt_dir[0], 
                       images_list=train_imgs,
                       size=val_size, outSize=outSize,
                       imgSize=imgSize, flip=False, fxtSize=valFxtSize),
        batch_size=ref_batch_size,
        shuffle=False,
        num_workers=numWorkers
    )

    print_str += '*Stats* training img number:{}, val img number:{}, ref img number:{}, batch size:{}'.format(len(train_loader.dataset.imgs), len(val_loader.dataset.imgs), ref_og_size, args.batch_size)
    return (train_loader, val_loader, ref_loader, print_str)

def create_nshot_merge_loaders(args, 
                        num_shots=5,
                        split_file='',
                        train_size=None, 
                        val_size=None, 
                        outSize=None, 
                        imgSize=None,
                        trFxtSize=None,
                        valFxtSize=None,
                        ref_batch_size=1,
                        shuffle=True,
                        flip=True,
                        numWorkers=4):
    shuffle_opt = shuffle
    if num_shots < ref_batch_size:
        ref_batch_size = num_shots

    if os.path.isfile(split_file):
        npzfile = np.load(split_file)
        train_imgs = npzfile['train_imgs']
        val_imgs = npzfile['val_imgs']
        train_imgs = [x for x in train_imgs.tolist()]
        val_imgs = [x for x in val_imgs.tolist()]
    else:
        image_names = load_allimages_wopath(args.val_img_dir[0])
        random.shuffle(image_names)
        train_imgs, val_imgs = image_names[:num_shots], image_names[num_shots:]

        np.savez(split_file, train_imgs=np.array(train_imgs), val_imgs=np.array(val_imgs))

    train_ref_imgs,train_gts = [],[]
    train_ref_imgs += [os.path.join(args.train_img_dir[0],x) for x in load_allimages_wopath(os.path.join(args.train_img_dir[0]))]
    train_gts += [os.path.join(args.train_gt_dir[0],x) for x in load_allimages_wopath(os.path.join(args.train_gt_dir[0]))]
    train_ref_imgs += [os.path.join(args.val_img_dir[0],x) for x in train_imgs]
    train_gts += [os.path.join(args.val_gt_dir[0],x) for x in train_imgs]
    data_source = NShotDataset_with_path(images_list=train_ref_imgs,
                   fxts_list=train_gts, 
                   size=train_size, outSize=outSize,
                   imgSize=imgSize, flip=False, fxtSize=trFxtSize)
    custom_sampler = None
    train_loader = torch.utils.data.DataLoader(
        data_source,
        batch_size=args.batch_size,
        shuffle=shuffle_opt,
        sampler=custom_sampler,
        num_workers=numWorkers
    )

    val_loader = torch.utils.data.DataLoader(
        NShotDataset(args.val_img_dir[0],
                       args.val_gt_dir[0], 
                       images_list=val_imgs,
                       size=val_size, outSize=outSize,
                       imgSize=imgSize, flip=False, fxtSize=valFxtSize),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=numWorkers
    )
    print_str = ''
    if len(train_imgs) <= 10:
        print_str += '-----training images------\n'
        print_str += ','.join(train_imgs)
        print_str += '\n--------------------------\n'
    train_imgs = np.array(train_imgs)
    ref_og_size = train_imgs.shape[0]
    num_batches = int(np.ceil(data_source.__len__()/args.batch_size))
    sample_ints = [np.random.randint(train_imgs.shape[0], size=ref_batch_size) for i in range(num_batches)]
    sample_ints = np.concatenate(sample_ints)
    train_imgs = train_imgs[sample_ints]
    ref_loader = torch.utils.data.DataLoader(
        NShotDataset(args.val_img_dir[0],
                       args.val_gt_dir[0], 
                       images_list=train_imgs,
                       size=val_size, outSize=outSize,
                       imgSize=imgSize, flip=False, fxtSize=valFxtSize),
        batch_size=ref_batch_size,
        shuffle=False,
        num_workers=numWorkers
    )

    print_str += '*Stats* training img number:{}, val img number:{}, ref img number:{}, batch size:{}'.format(len(train_loader.dataset.imgs), len(val_loader.dataset.imgs), ref_og_size, args.batch_size)
    return (train_loader, val_loader, ref_loader, print_str)

def create_loaders_fromsplit(args,
                        split_file='',
                        train_size=None, 
                        val_size=None, 
                        outSize=None, 
                        imgSize=None,
                        trFxtSize=None,
                        valFxtSize=None,
                        shuffle=True,
                        flip=True,
                        numWorkers=4):
    shuffle_opt = shuffle
    custom_sampler = None

    npzfile = np.load(split_file)
    train_imgs = npzfile['train_imgs']
    val_imgs = npzfile['val_imgs']
    train_imgs = [x for x in train_imgs.tolist()]
    val_imgs = [x for x in val_imgs.tolist()]

    if len(train_imgs) <= 10:
        print('-----training images------')
        print(train_imgs)
        print('--------------------------')

    val_loader = torch.utils.data.DataLoader(
        NShotDataset(args.val_img_dir[0],
                       args.val_gt_dir[0], 
                       images_list=val_imgs,
                       size=val_size, outSize=outSize,
                       imgSize=imgSize, flip=False, fxtSize=valFxtSize),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=numWorkers
    )

    train_loader = torch.utils.data.DataLoader(
        NShotDataset(args.val_img_dir[0],
                       args.val_gt_dir[0], 
                       images_list=train_imgs,
                       size=val_size, outSize=outSize,
                       imgSize=imgSize, flip=False, fxtSize=valFxtSize),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=numWorkers
    )

    print('*Stats* training img number:{}, val img number:{}, batch size:{}'.format(len(train_loader.dataset.imgs), len(val_loader.dataset.imgs), args.batch_size))
    return (train_loader, val_loader)

def create_nshotsplit_loaders(args, 
                        num_shots=5,
                        split_file='',
                        train_size=None, 
                        val_size=None, 
                        outSize=None, 
                        imgSize=None,
                        trFxtSize=None,
                        valFxtSize=None,
                        shuffle=True,
                        flip=True,
                        numWorkers=4):
    shuffle_opt = shuffle
    custom_sampler = None

    if os.path.isfile(split_file):
        npzfile = np.load(split_file)
        train_imgs = npzfile['train_imgs']
        val_imgs = npzfile['val_imgs']
        train_imgs = [x for x in train_imgs.tolist()]
        val_imgs = [x for x in val_imgs.tolist()]
    else:
        image_names = load_allimages_wopath(args.val_img_dir[0])
        random.shuffle(image_names)
        train_imgs, val_imgs = image_names[:num_shots], image_names[num_shots:]

        np.savez(split_file, train_imgs=np.array(train_imgs), val_imgs=np.array(val_imgs))

    print_str = ''
    if len(train_imgs) <= 10:
        print_str += '-----training images------\n'
        print_str += ','.join(train_imgs)
        print_str += '\n--------------------------\n'

    val_loader = torch.utils.data.DataLoader(
        NShotDataset(args.val_img_dir[0],
                       args.val_gt_dir[0], 
                       images_list=val_imgs,
                       size=val_size, outSize=outSize,
                       imgSize=imgSize, flip=False, fxtSize=valFxtSize),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=numWorkers
    )

    train_loader = torch.utils.data.DataLoader(
        NShotDataset(args.val_img_dir[0],
                       args.val_gt_dir[0], 
                       images_list=train_imgs,
                       size=val_size, outSize=outSize,
                       imgSize=imgSize, flip=False, fxtSize=valFxtSize),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=numWorkers
    )

    print_str += '*Stats* training img number:{}, val img number:{}, batch size:{}'.format(len(train_loader.dataset.imgs), len(val_loader.dataset.imgs), args.batch_size)
    return (train_loader, val_loader, print_str)

def create_train_data_loader(args, 
                        _size=None, 
                        outSize=None, 
                        imgSize=None,
                        batchSize=1,
                        shuffle=True,
                        flip=False,
                        numWorkers=4):
    train_loader = torch.utils.data.DataLoader(
        SaliconDataset('%s' % args.train_img_dir,
                       '%s' % args.train_gt_dir, 
                       size=_size, 
                       flip=flip,
                       outSize=outSize,
                       imgSize=imgSize),
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=numWorkers
    )
    return train_loader

def create_val_data_loader(val_img_dir, val_gt_dir, val_batch_size, imgSize, numWorkers=4):

    val_loader = torch.utils.data.DataLoader(
        SaliconDataset('{}'.format(val_img_dir),
                       '{}'.format(val_gt_dir), 
                       imgSize=imgSize, flip=False),
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=numWorkers
    )
    print('*Stats* val img number:{}, batch size:{}'.format(len(val_loader.dataset.imgs), val_batch_size))
    return val_loader


def create_test_data_loader(img_dir, 
                            batch_size, 
                            imgSize, 
                            numWorkers=4):

    test_loader = torch.utils.data.DataLoader(
        SaliconDataset_nGT('{}'.format(img_dir),
                       imgSize=imgSize, flip=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=numWorkers
    )
    print('*Stats* test img number:{}, batch size:{}'.format(len(test_loader.dataset.imgs), batch_size))
    return test_loader

def train_val(model, criterion, optimizer, epoch,
                 train_loader, val_loader, sal_path, _logger=None):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_losses = AverageMeter()
    val_losses = AverageMeter()

    train_sim = AverageMeter()
    val_sim = AverageMeter()

    train_cc = AverageMeter()
    val_cc = AverageMeter()

    train_cos_before = AverageMeter()
    train_cos_after = AverageMeter()
    train_mag = AverageMeter()

    train_num = 0
    val_num = 0
    val_iter = iter(val_loader)

    nCountImg = 0
    sal_path = '{}/ep{}'.format(sal_path, epoch+1)
    ensure_dir(sal_path)  

    # =============start to train==============
    model.train()
    model.apply(set_bn_eval)

    t = 0
    cur_lr = optimizer.param_groups[0]['lr']
    j = 0
    end = time.time()
    time_begin = end
    for i, (X, Y, pathnames) in enumerate(train_loader):
        X = Variable(X).cuda()
        if type(Y).__name__=='list':
            Y = [Variable(Y[i], requires_grad=False).cuda() for i in range(len(Y))]
            orgsz = Y[0].size()
        else:
            Y = Variable(Y, requires_grad=False).cuda()
            orgsz = Y.size()
        train_num += orgsz[0]

        # measure data loading time
        data_time.update(time.time() - end)

        x_output = model(X)
        
        if x_output.shape[2] != Y.shape[2] or x_output.shape[3] != Y.shape[3]:
            x_output = F.interpolate(x_output,size=Y.size()[2:], mode='bilinear', align_corners=True)


        loss = criterion(x_output, Y)
        sal_sz = x_output.size()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cc_score = 0

        train_losses.update(loss.item(), X.size(0))

        batch_time.update(time.time() - end)
        
        end = time.time()
        
        
    totaltime = (end-time_begin) # /60 convert to minutes
    print('Train [{0}]: [{1}/{2}]\t'
        'LearningRate {3:.6f}\t'
        'Time {4:.3f} ({batch_time.avg:.3f})\t'
        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.8f})'.format(
        epoch+1, i+1, train_num, cur_lr, totaltime, batch_time=batch_time,
        data_time=data_time, loss=train_losses))
    train_datatime = data_time.avg
    train_batchtime = batch_time.avg
    if _logger is not None:
        print('Train [{0}]: [{1}/{2}]\t'
            'LearningRate {3:.6f}\t'
            'Time {4:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.8f})'.format(
            epoch+1, i+1, train_num, cur_lr, totaltime, batch_time=batch_time,
            data_time=data_time, loss=train_losses),
            file=_logger,flush=True)

    # =============switch to evaluation mode==============
    batch_time.reset()
    data_time.reset()
    model.eval()
    time_begin = time.time()
    for k in range(len(val_loader)):

        end = time.time()
        
        X, Y, pathnames = val_iter.next()
        X = Variable(X).cuda()
        Y = Variable(Y).cuda()
        orgsz = Y.size()
        val_num += orgsz[0]
        filenames = ['{}'.format(element[element.rfind('/')+1:]) for element in pathnames]

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(X)

        if output.shape[2] != Y.shape[2] or output.shape[3] != Y.shape[3]:
            output = F.interpolate(output,size=Y.size()[2:], mode='bilinear', align_corners=True)

        loss = criterion(output, Y)
        sal_sz = output.size()

        # record loss
        val_losses.update(loss.item(), X.size(0))

        salmap = output.view(sal_sz[0],1,sal_sz[2],sal_sz[3])
        for i_sal in range(salmap.size()[0]):
            nCountImg = nCountImg+1
            filename = filenames[i_sal]
            sqz_salmap = salmap[i_sal].squeeze()
            sqz_salmap = sqz_salmap.data
            sqz_salmap = sqz_salmap - sqz_salmap.min()
            sqz_salmap = sqz_salmap / sqz_salmap.max()
            cur_save_path = os.path.join(sal_path, filename)
            sqz_salmap = sqz_salmap.cpu().numpy()
            sqz_salmap *= 255.0
            sqz_salmap = sqz_salmap.astype(np.uint8)
            sqz_salmap = Image.fromarray(sqz_salmap)
            sqz_salmap.save(cur_save_path)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        j += 1
    totaltime = (end-time_begin) # /60 convert to minutes
    print('Test [{0}]: [{1}/{2}]\t'
          'Time {3:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.8f})'.format(
           epoch+1, j, val_num, totaltime, batch_time=batch_time,
           data_time=data_time, loss=val_losses))
    if _logger is not None:
        print('Test [{0}]: [{1}/{2}]\t'
              'Time {3:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.8f})'.format(
               epoch+1, j, val_num, totaltime, batch_time=batch_time,
               data_time=data_time, loss=val_losses),
              file=_logger,flush=True)
    val_datatime = data_time.avg
    val_batchtime = batch_time.avg
    # switch to training mode
    model.train(True)
    # =============end of evaluation mode==============

    cur_epoch = epoch+1
    return (train_losses.avg, val_losses.avg, train_batchtime, train_datatime, val_batchtime, val_datatime)

def train_val_nshot(model, criterion, optimizer, epoch,
                 train_loader, val_loader, sal_paths, _logger=None):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_losses = AverageMeter()
    val_losses = AverageMeter()
    train_losses1 = AverageMeter()
    train_losses2 = AverageMeter()
    val_losses1 = AverageMeter()
    val_losses2 = AverageMeter()

    train_sim = AverageMeter()
    val_sim = AverageMeter()

    train_cc = AverageMeter()
    val_cc = AverageMeter()

    train_cos_before = AverageMeter()
    train_cos_after = AverageMeter()
    train_mag = AverageMeter()

    train_num = 0
    val_num = 0
    val_iter = iter(val_loader)

    nCountImg = 0
    ep_sal_path = sal_paths[0]
    ensure_dir(ep_sal_path)
    if len(sal_paths) == 2:
        rec_sal_path = sal_paths[1]
        ensure_dir(rec_sal_path)

    # =============start to train==============
    # switch to training mode
    model.train()
    model.apply(set_bn_eval)

    t = 0
    cur_lr = optimizer.param_groups[0]['lr']
    j = 0
    end = time.time()
    time_begin = end
    for i, (X, Y, pathnames) in enumerate(train_loader):
        X = Variable(X).cuda()
        if type(Y).__name__=='list':
            Y = [Variable(Y[i], requires_grad=False).cuda() for i in range(len(Y))]
            orgsz = Y[0].size()
        else:
            Y = Variable(Y, requires_grad=False).cuda()
            orgsz = Y.size()
        
        train_num += orgsz[0]

        # measure data loading time
        data_time.update(time.time() - end)

        batch_start_time = time.time()
        # compute output
        x_output = model(X)
        
        if x_output.shape[2] != Y.shape[2] or x_output.shape[3] != Y.shape[3]:
            x_output = F.interpolate(x_output,size=Y.size()[2:], mode='bilinear', align_corners=True)

        loss = criterion(x_output, Y)
        sal_sz = x_output.size()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - batch_start_time)

        # record loss
        train_losses.update(loss.item(), X.size(0))
        
        end = time.time()
        
        
    totaltime = (end-time_begin) # /60 convert to minutes
    print('Train [{0}]: [{1}/{2}]\t'
        'LearningRate {3:.6f}\t'
        'Time {4:.3f} ({batch_time.avg:.3f})\t'
        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.8f})'.format(
        epoch+1, i+1, train_num, cur_lr, totaltime, batch_time=batch_time,
        data_time=data_time, loss=train_losses))

    train_datatime = data_time.avg
    train_batchtime = batch_time.avg

    if _logger is not None:
        print('Train [{0}]: [{1}/{2}]\t'
            'LearningRate {3:.6f}\t'
            'Time {4:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.8f})'.format(
            epoch+1, i+1, train_num, cur_lr, totaltime, batch_time=batch_time,
            data_time=data_time, loss=train_losses),
            file=_logger,flush=True)
    # =============switch to evaluation mode==============
    batch_time.reset()
    data_time.reset()
    model.eval()
    time_begin = time.time()
    for k in range(len(val_loader)):

        end = time.time()
        
        X, Y, pathnames = val_iter.next()
        X = Variable(X).cuda()
        Y = Variable(Y).cuda()
        orgsz = Y.size()
        val_num += orgsz[0]
        filenames = ['{}'.format(element[element.rfind('/')+1:]) for element in pathnames]

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(X)

        if output.shape[2] != Y.shape[2] or output.shape[3] != Y.shape[3]:
            output = F.interpolate(output,size=Y.size()[2:], mode='bilinear', align_corners=True)

        loss = criterion(output, Y)
        sal_sz = output.size()

        # record loss
        val_losses.update(loss.item(), X.size(0))
        
        # predict salmaps
        salmap = output.view(sal_sz[0],1,sal_sz[2],sal_sz[3])
        for i_sal in range(salmap.size()[0]):
            nCountImg = nCountImg+1
            filename = filenames[i_sal]
            sqz_salmap = salmap[i_sal].squeeze()
            sqz_salmap = sqz_salmap.data
            sqz_salmap = sqz_salmap - sqz_salmap.min()
            sqz_salmap = sqz_salmap / sqz_salmap.max()
            cur_save_path = os.path.join(ep_sal_path, filename)
            sqz_salmap = sqz_salmap.cpu().numpy()
            sqz_salmap *= 255.0
            sqz_salmap = sqz_salmap.astype(np.uint8)
            sqz_salmap = Image.fromarray(sqz_salmap)
            sqz_salmap.save(cur_save_path)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        j += 1
    totaltime = (end-time_begin) # /60 convert to minutes
    print('Test [{0}]: [{1}/{2}]\t'
            'Time {3:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.8f})'.format(
           epoch+1, j, val_num, totaltime, batch_time=batch_time,
           data_time=data_time, loss=val_losses))
    if _logger is not None:
        print('Test [{0}]: [{1}/{2}]\t'
                'Time {3:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.8f})'.format(
               epoch+1, j, val_num, totaltime, batch_time=batch_time,
               data_time=data_time, loss=val_losses),
              file=_logger,flush=True)
    val_datatime = data_time.avg
    val_batchtime = batch_time.avg
    # switch to training mode
    model.train(True)
    # =============end of evaluation mode==============

    cur_epoch = epoch+1
    return (train_losses.avg, val_losses.avg, train_batchtime, train_datatime, val_batchtime, val_datatime)

def train_val_ref(model, model_head, criterion, optimizer, optimizer_head, epoch,
                 train_loader, val_loader, ref_loader, sal_paths, _logger=None):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_losses = AverageMeter()
    val_losses = AverageMeter()
    train_losses1 = AverageMeter()
    train_losses2 = AverageMeter()
    val_losses1 = AverageMeter()
    val_losses2 = AverageMeter()

    train_sim = AverageMeter()
    val_sim = AverageMeter()

    bef_cos_sim = AverageMeter()
    aft_cos_sim = AverageMeter()

    train_cc = AverageMeter()
    val_cc = AverageMeter()

    train_cos_before = AverageMeter()
    train_cos_after = AverageMeter()
    train_mag = AverageMeter()

    train_num = 0
    val_num = 0
    val_iter = iter(val_loader)

    nCountImg = 0
    ep_sal_path = sal_paths[0]
    ensure_dir(ep_sal_path)

    # =============start to train==============
    # switch to training mode
    model.train()
    model.apply(set_bn_eval)

    t = 0
    cur_lr = optimizer.param_groups[0]['lr']
    j = 0
    end = time.time()
    time_begin = end
    for i, ((X, Y, pathnames), (X_ref, Y_ref, path_ref)) in enumerate(zip(train_loader, ref_loader)):
        X = Variable(X).cuda()
        Y = Variable(Y, requires_grad=False).cuda()
        orgsz = Y.size()
        X_ref = Variable(X_ref).cuda()
        Y_ref = Variable(Y_ref, requires_grad=False).cuda()
        
        train_num += orgsz[0]

        # measure data loading time
        data_time.update(time.time() - end)

        batch_start_time = time.time()
        # compute output
        with torch.no_grad():
            x_ref_out = model(X_ref)
        x_output = model(X)
        x_output, loss, cos_sim = model_head.refer_and_learn(x_output, Y, x_ref_out, Y_ref, criterion, optimizer, optimizer_head)
        batch_time.update(time.time() - batch_start_time)
        sal_sz = x_output.size()

        # record loss
        train_losses.update(loss.item(), X.size(0))
        bef_cos_sim.update(cos_sim[0], X.size(0))
        aft_cos_sim.update(cos_sim[1], X.size(0))
        
        end = time.time()
        
        
    totaltime = (end-time_begin) # /60 convert to minutes
    print('Train [{0}]: [{1}/{2}]\t'
        'LearningRate {3:.6f}\t'
        'Time {4:.3f} ({batch_time.avg:.3f})\t'
        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.8f})\t'
        'bef_cos {bef_cos.avg:.4f}\t'
        'aft_cos {aft_cos.avg:.4f}'.format(
        epoch+1, i+1, train_num, cur_lr, totaltime, batch_time=batch_time,
        data_time=data_time, loss=train_losses,
        bef_cos=bef_cos_sim, aft_cos=aft_cos_sim))

    train_datatime = data_time.avg
    train_batchtime = batch_time.avg

    if _logger is not None:
        print('Train [{0}]: [{1}/{2}]\t'
            'LearningRate {3:.6f}\t'
            'Time {4:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.8f})\t'
            'bef_cos {bef_cos.avg:.4f}\t'
            'aft_cos {aft_cos.avg:.4f}'.format(
            epoch+1, i+1, train_num, cur_lr, totaltime, batch_time=batch_time,
            data_time=data_time, loss=train_losses, bef_cos=bef_cos_sim, aft_cos=aft_cos_sim),
            file=_logger,flush=True)
    # =============switch to evaluation mode==============
    batch_time.reset()
    data_time.reset()
    model.eval()
    time_begin = time.time()
    for k in range(len(val_loader)):

        end = time.time()
        
        X, Y, pathnames = val_iter.next()
        X = Variable(X).cuda()
        Y = Variable(Y).cuda()
        orgsz = Y.size()
        val_num += orgsz[0]
        filenames = ['{}'.format(element[element.rfind('/')+1:]) for element in pathnames]

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(X)
        output = model_head(output)

        if output.shape[2] != Y.shape[2] or output.shape[3] != Y.shape[3]:
            output = F.interpolate(output,size=Y.size()[2:], mode='bilinear', align_corners=True)

        loss = criterion(output, Y)
        sal_sz = output.size()

        # record loss
        val_losses.update(loss.item(), X.size(0))

        # predict salmaps
        salmap = output.view(sal_sz[0],1,sal_sz[2],sal_sz[3])
        for i_sal in range(salmap.size()[0]):
            nCountImg = nCountImg+1
            filename = filenames[i_sal]
            sqz_salmap = salmap[i_sal].squeeze()
            sqz_salmap = sqz_salmap.data
            sqz_salmap = sqz_salmap - sqz_salmap.min()
            sqz_salmap = sqz_salmap / sqz_salmap.max()
            cur_save_path = os.path.join(ep_sal_path, filename)
            sqz_salmap = sqz_salmap.cpu().numpy()
            sqz_salmap *= 255.0
            sqz_salmap = sqz_salmap.astype(np.uint8)
            sqz_salmap = Image.fromarray(sqz_salmap)
            sqz_salmap.save(cur_save_path)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        j += 1
    totaltime = (end-time_begin) # /60 convert to minutes
    print('Test [{0}]: [{1}/{2}]\t'
            'Time {3:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.8f})'.format(
           epoch+1, j, val_num, totaltime, batch_time=batch_time,
           data_time=data_time, loss=val_losses))
    if _logger is not None:
        print('Test [{0}]: [{1}/{2}]\t'
                'Time {3:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.8f})'.format(
               epoch+1, j, val_num, totaltime, batch_time=batch_time,
               data_time=data_time, loss=val_losses),
              file=_logger,flush=True)
    val_datatime = data_time.avg
    val_batchtime = batch_time.avg
    # switch to training mode
    model.train(True)
    # =============end of evaluation mode==============

    cur_epoch = epoch+1
    return (train_losses.avg, val_losses.avg, train_batchtime, train_datatime, val_batchtime, val_datatime)

def predict(model, 
            val_loader, 
            sal_path,
            sigma=-1.0,
            truncate=4.0,
            file_type='jpg'):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_num = 0
    val_num = 0
    val_iter = iter(val_loader)

    nCountImg = 0
    ensure_dir(sal_path)  

    mem = None
    fnames = []
    allgrads = None
    inProducts = []
    cosines = []
    losses = []
    model.eval()


    time_begin = time.time()
    
    for i, (X, pathnames, img_size) in enumerate(val_loader):
        if i%500 == 0:
            print('processing {}-th sample'.format(i+1))
        end = time.time()
        filenames = ['{}'.format(element[element.rfind('/')+1:]) for element in pathnames]
        X = Variable(X).cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(X)

        if output.shape[2] != img_size[0] or output.shape[3] != img_size[1]:
            output = F.interpolate(output,size=img_size, mode='bilinear', align_corners=True) #, align_corners=True
        
        sal_sz = output.size()

        salmap = output.view(sal_sz[0],1,sal_sz[2],sal_sz[3])
        for i_sal in range(salmap.size()[0]):
            nCountImg = nCountImg+1
            filename = filenames[i_sal]
            sqz_salmap = salmap[i_sal].unsqueeze(0)
            if sigma > 0:
                w_s = 2*int(truncate*sigma + 0.5) + 1
                sqz_salmap = kornia.gaussian_blur(sqz_salmap, (w_s, w_s), (sigma, sigma))
            sqz_salmap = sqz_salmap.squeeze()
            sqz_salmap = sqz_salmap.data
            sqz_salmap = sqz_salmap - sqz_salmap.min()
            sqz_salmap = sqz_salmap / sqz_salmap.max()

            cur_save_path = os.path.join(sal_path, filename[:filename.index('.')+1]+file_type)

            sqz_salmap = sqz_salmap.cpu().numpy()
            sqz_salmap *= 255.0
            sqz_salmap = sqz_salmap.astype(np.uint8)
            sqz_salmap = Image.fromarray(sqz_salmap)
            sqz_salmap.save(cur_save_path)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    totaltime = (end-time_begin) # /60 convert to minutes
    
    val_datatime = data_time.avg
    val_batchtime = batch_time.avg
    return (val_batchtime, val_datatime)

def save_checkpoint(model, save_file):
    if model.__class__.__name__!='DataParallel':
        torch.save(model.state_dict(), save_file)
    else:
        torch.save(model.module.state_dict(), save_file)

def adjust_learning_rate(lr, optimizer, epoch, basenum=2, coef=0.1):
    lr = lr * coef ** int(epoch/basenum)
    
    ncount = 1
    for param_group in optimizer.param_groups:
        
        if lr >= 1e-7:
            param_group['lr'] = lr
        ncount = ncount+1
    
    return lr

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_list(alist):
    return sum(p.numel() for p in alist if p.requires_grad)

def getParameters(model):
    cond_param_list = []
    param_list = []
    for name, param in model.named_parameters():
        if not 'condition' in name:
            yield param