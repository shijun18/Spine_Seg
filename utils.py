import os
import h5py
import SimpleITK as sitk
import numpy as np


def nii_reader(data_path):
    """
    Convert input data to numpy array

    returns:
    :info, dict, information about the input data, including inplane size, pixel-spacing and thickness
    :image, numpy array
    """
    info = {}
    meta_data = sitk.ReadImage(data_path)
    image = sitk.GetArrayFromImage(meta_data).astype(np.float32)
    # print("max:",np.max(image))
    # print("min:",np.min(image))

    info['spacing'] = meta_data.GetSpacing()
    info['origin'] = meta_data.GetOrigin()
    info['direction'] = meta_data.GetDirection()
    info['inplane_size'] = tuple(meta_data.GetSize()[:2])
    info['pixel_spacing'] = tuple(meta_data.GetSpacing()[:2])
    info['thickness'] = meta_data.GetSpacing()[-1]
    info['z_size'] = meta_data.GetSize()[-1]

    return info, image

def save_as_nii(data, save_path, info):
    sitk_data = sitk.GetImageFromArray(data)
    sitk_data.SetSpacing(info['spacing'])
    sitk_data.SetOrigin(info['origin'])
    sitk_data.SetDirection(info['direction'])

    sitk.WriteImage(sitk_data, save_path)


def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()


def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image



def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        print(pth_list)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        return None
    
    

def remove_weight_path(ckpt_path,retain=1):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path,pth_item))


def dfs_remove_weight(ckpt_path):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path)
        else:
            remove_weight_path(ckpt_path)
            break  

if __name__ == "__main__":

    ckpt_path = './ckpt/'
    dfs_remove_weight(ckpt_path)