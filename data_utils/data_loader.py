from numpy.lib.function_base import append
from config import PATH_LIST
from os import path
import sys
sys.path.append('..')

from torch.utils.data import Dataset
import torch
import numpy as np
import random
from utils import hdf5_reader
from skimage.transform import resize
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter



def data_pre_loader(data_path,resize=False,dim=None,num_class=None):
    image = []
    label = []
    for item in data_path:
        img = hdf5_reader(item,'image')
        lab = hdf5_reader(item,'label')
        if resize:
            if dim is not None and img.shape != dim:
                img = resize(img, dim, anti_aliasing=True)
                temp_lab = np.zeros(dim,dtype=np.float32)
                for z in range(1, num_class):
                    roi = resize((lab == z).astype(np.float32),dim,mode='constant')
                    temp_lab[roi >= 0.5] = z
                lab = temp_lab
            
        image.append(img)
        label.append(lab)

    return image,label


class Normalize(object):
    '''
    normalize image to 0-mean and 1-std.
    '''
    def __init__(self, mean, std, eps=1e-4):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        # gray truncation
        image = image - self.mean
        image = image / (self.std + self.eps)

        new_sample = {'image': image, 'mask': mask}

        return new_sample


class Trunc_and_Normalize(object):
    '''
    truncate gray scale and normalize to [0,1]
    '''
    def __init__(self, scale):
        self.scale = scale
        assert len(self.scale) == 2, 'scale error'

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        # gray truncation
        image = image - self.scale[0]
        gray_range = self.scale[1] - self.scale[0]
        image[image < 0] = 0
        image[image > gray_range] = gray_range

        image = image / gray_range

        new_sample = {'image': image, 'mask': mask}

        return new_sample


class CropResize(object):
    '''
    Data preprocessing.
    Adjust the size of input data to fixed size by cropping and resize
    Args:
    - dim: tuple of integer, fixed size
    - crop: single integer, factor of cropping, H/W ->[:,crop:-crop,crop:-crop]
    '''
    def __init__(self, dim=None, num_class=2, crop=0):
        self.dim = dim
        self.num_class = num_class
        self.crop = crop

    def __call__(self, sample):

        # image: numpy array
        # mask: numpy array
        image = sample['image']
        mask = sample['mask']
        # crop
        if self.crop != 0:
            image = image[self.crop:-self.crop, self.crop:-self.crop]
            mask = mask[self.crop:-self.crop, self.crop:-self.crop]
        # resize
        if self.dim is not None and image.shape != self.dim:
            image = resize(image, self.dim, anti_aliasing=True)
            temp_mask = np.zeros(self.dim,dtype=np.float32)
            for z in range(1, self.num_class):
                roi = resize((mask == z).astype(np.float32),self.dim,mode='constant')
                temp_mask[roi >= 0.5] = z
            mask = temp_mask

        new_sample = {'image': image, 'mask': mask}

        return new_sample


class To_Tensor(object):
    '''
    Convert the data in sample to torch Tensor.
    Args:
    - n_class: the number of class
    '''
    def __init__(self, num_class=2):
        self.num_class = num_class

    def __call__(self, sample):

        image = sample['image']
        mask = sample['mask']
        # expand dims

        new_image = np.expand_dims(image, axis=0)
        new_mask = np.empty((self.num_class, ) + mask.shape, dtype=np.float32)
        for z in range(1,self.num_class):
            temp = (mask == z).astype(np.float32)
            new_mask[z, ...] = temp
        new_mask[0,...] = np.amax(new_mask[1:, ...],axis=0) == 0
        # convert to Tensor
        new_sample = {
            'image': torch.from_numpy(new_image),
            'mask': torch.from_numpy(new_mask)
        }

        return new_sample


class DataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of image path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''
    def __init__(self,
                 path_list=None,
                 roi_number=None,
                 num_class=2,
                 input_shape=None,
                 transform=None,
                 prefetch=False):

        self.path_list = path_list
        self.roi_number = roi_number
        self.num_class = num_class
        self.input_shape = input_shape
        self.transform = transform
        self.prefetch = prefetch

        if self.prefetch:
            self.image_buffer, self.mask_buffer = data_pre_loader(self.path_list,False,self.input_shape,self.num_class)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        # Get image and mask
        if self.prefetch:
            image = self.image_buffer[index]
            mask = self.mask_buffer[index]
        else:
            image = hdf5_reader(self.path_list[index],'image')
            mask = hdf5_reader(self.path_list[index],'label')

        if self.roi_number is not None:
            if isinstance(self.roi_number,list):
                tmp_mask = np.zeros_like(mask,dtype=np.float32)
                assert self.num_class == len(self.roi_number) + 1
                for i, roi in enumerate(self.roi_number):
                    tmp_mask[mask == roi] = i+1
                mask = tmp_mask
            else:
                assert self.num_class == 2
                mask = (mask == self.roi_number).astype(np.float32)

        sample = {'image': image, 'mask': mask}
        if self.transform is not None:
            sample = self.transform(sample)

        label = np.zeros((self.num_class, ), dtype=np.float32)
        label_array = np.argmax(sample['mask'].numpy(),axis=0)
        label[np.unique(label_array).astype(np.uint8)] = 1

        sample['label'] = torch.Tensor(list(label[1:]))

        return sample


class BalanceDataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of two lists, one includes positive samples, and the other includes negative samples
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''
    def __init__(self,
                 path_list=None,
                 roi_number=None,
                 num_class=2,
                 input_shape=None,
                 transform=None,
                 prefetch=False,
                 factor=0.3):

        self.path_list = path_list
        self.roi_number = roi_number
        self.num_class = num_class
        self.input_shape = input_shape
        self.transform = transform
        self.prefetch = prefetch
        self.factor = factor

        if self.prefetch:
            self.image_buffer = []
            self.mask_buffer = []
            for sublist in self.path_list:
                tmp_image, tmp_mask = data_pre_loader(sublist,False,self.input_shape,self.num_class)
                self.image_buffer.append(tmp_image)
                self.mask_buffer.append(tmp_mask)

    def __len__(self):
        assert isinstance(self.path_list[0],list)
        assert len(self.path_list) == 2
        return sum([len(case) for case in self.path_list])

    def __getitem__(self, index):
        # balance sampler
        if self.prefetch:
            list_id = int(random.random() < self.factor)
            index = random.choice(range(len(self.path_list[list_id])))
            image = self.image_buffer[list_id][index]
            mask = self.mask_buffer[list_id][index]
        else:
            item_path = random.choice(self.path_list[int(random.random() < self.factor)])
            # Get image and mask
            image = hdf5_reader(item_path,'image')
            mask = hdf5_reader(item_path,'label')

        if self.roi_number is not None:
            if isinstance(self.roi_number,list):
                tmp_mask = np.zeros_like(mask,dtype=np.float32)
                assert self.num_class == len(self.roi_number) + 1
                for i, roi in enumerate(self.roi_number):
                    tmp_mask[mask == roi] = i+1
                mask = tmp_mask
            else:
                assert self.num_class == 2
                mask = (mask == self.roi_number).astype(np.float32)

        sample = {'image': image, 'mask': mask}
        if self.transform is not None:
            sample = self.transform(sample)

        label = np.zeros((self.num_class, ), dtype=np.float32)
        label_array = np.argmax(sample['mask'].numpy(),axis=0)
        label[np.unique(label_array).astype(np.uint8)] = 1

        sample['label'] = torch.Tensor(list(label[1:]))

        return sample