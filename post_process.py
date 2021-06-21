import sys
import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm



def post_process(label,n):
    from skimage.morphology import remove_small_objects

    final= np.zeros_like(label,dtype=np.uint8)
    for i in range(1,n):
        roi = (label == i).astype(np.bool)
        roi = remove_small_objects(roi,min_size=64, connectivity=1,in_place=False)
        final[roi == 1] = i
    return final


# result_path = './result/tmp/fold5'
# result_path = './result/Spine/v1-2-4.1-all/All/fusion/'
result_path = './result/Spine/final/fusion/'
# save_folder = './result/tmp/post_fold5'
# save_folder = './result/Spine/v1-2-4.1-all/All/post_fusion/'
save_folder = './result/Spine/final/post_fusion/'


if not os.path.exists(save_folder):
    os.makedirs(save_folder)


for item in tqdm(os.scandir(result_path)):
    data = sitk.ReadImage(item.path)
    label = sitk.GetArrayFromImage(data).astype(np.uint8)
    
    spacing = data.GetSpacing()
    origin = data.GetOrigin()
    direction = data.GetDirection()

    for i in range(label.shape[0]):
        label[i] = post_process(label[i],20)

    sitk_data = sitk.GetImageFromArray(label)
    sitk_data.SetSpacing(spacing)
    sitk_data.SetOrigin(origin)
    sitk_data.SetDirection(direction)
    
    save_path = os.path.join(save_folder,item.name)
    sitk.WriteImage(sitk_data, save_path)
