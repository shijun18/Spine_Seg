import numpy as np
import h5py
import SimpleITK as sitk

import glob
import os
import pydicom



def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()


def save_as_nii(data, save_path):
    sitk_data = sitk.GetImageFromArray(data)
    sitk.WriteImage(sitk_data, save_path)



## nii.gz reader
def nii_reader(data_path):
    data = sitk.ReadImage(data_path)
    image = sitk.GetArrayFromImage(data).astype(np.float32)
    return data,image


def trunc_gray(img, in_range=(-1000, 600)):
    img = img - in_range[0]
    scale = in_range[1] - in_range[0]
    img[img < 0] = 0
    img[img > scale] = scale

    return img
    

def normalize(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return img

if __name__ == "__main__":
    data_path = '/staff/shijun/dataset/Nasopharynx_Oar/200301_CT'
    meta_data,image = dicom_series_reader(data_path)
    print(image.shape)
    print(meta_data[0].SliceThickness)