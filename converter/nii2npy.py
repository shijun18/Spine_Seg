import sys
sys.path.append('..')
import os
import glob
from tqdm import tqdm
import time
import shutil
import json
import numpy as np

from converter.nii_reader import Nii_Reader
from converter.utils import save_as_hdf5


# Different samples are saved in different folder
def nii_to_hdf5(input_path, save_path, annotation_list, target_format=None, resample=True):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    image_path = os.path.join(input_path,'MR')
    label_path = os.path.join(input_path,'Mask')
    start = time.time()
    for item in tqdm(os.scandir(image_path)):
        lab_path = os.path.join(label_path,'mask_' + item.name.lower())
        
        try:
            reader = Nii_Reader(item.path, target_format, lab_path, annotation_list, trunc_flag=False, normalize_flag=False)
        except:
            print("Error data: %s" % item.name.split('.')[0])
            continue
        else:
            if resample:
                images = reader.get_resample_images().astype(np.int16)
                labels = reader.get_resample_labels().astype(np.uint8)
            else:
                images = reader.get_raw_images().astype(np.int16)
                labels = reader.get_raw_labels().astype(np.uint8)

            hdf5_path = os.path.join(save_path, item.name.split('.')[0] + '.hdf5')

            save_as_hdf5(images, hdf5_path, 'image')
            save_as_hdf5(labels, hdf5_path, 'label')

    print("run time: %.3f" % (time.time() - start))



if __name__ == "__main__":

    json_file = './static_files/spine.json'
    with open(json_file, 'r') as fp:
        info = json.load(fp)
    nii_to_hdf5(info['nii_path'], info['npy_path'], info['annotation_list'],resample=False)