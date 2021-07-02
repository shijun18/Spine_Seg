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


# part10_path = './result/Spine/v4.3-all/Part_10/fusion/'
# part9_path = './result/Spine/v4.3-all/Part_9/fusion/'
# save_folder = './result/Spine/v4.3-all/segmentation_results/'

# part10_path = './result/Spine/v4.10-balance/Part_10/fusion/'
# part9_path = './result/Spine/v4.10-balance/Part_9/fusion/'
# save_folder = './result/Spine/v4.10-balance/segmentation_results/'

# part10_path = './result/Spine/v4.3-balance/Part_10/fusion/'
# part9_path = './result/Spine/v4.3-balance/Part_9/fusion/'
# save_folder = './result/Spine/v4.3-balance/segmentation_results/'

version_list = ['v4.3-balance','v4.3-all','v4.10-balance','final']

for ver in version_list[-1:]:
    # part10_path = './post_result/Spine/{}/Part_10/fusion/'.format(ver)
    # part9_path = './post_result/Spine/{}/Part_9/fusion/'.format(ver)
    # save_folder = './post_result/Spine/{}/segmentation_results/'.format(ver)

    part10_path = './result/Spine/{}/Part_10/fusion/'.format(ver)
    part9_path = './result/Spine/{}/Part_9/fusion/'.format(ver)
    save_folder = './result/Spine/{}/segmentation_results/'.format(ver)
    POST = True

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    for part10_item in tqdm(os.scandir(part10_path)):
        part10_data = sitk.ReadImage(part10_item.path)
        part10_label = sitk.GetArrayFromImage(part10_data).astype(np.uint8)
        
        spacing = part10_data.GetSpacing()
        origin = part10_data.GetOrigin()
        direction = part10_data.GetDirection()

        final_label = np.zeros_like(part10_label,dtype=np.uint8)

        part9_item_path = os.path.join(part9_path,part10_item.name)
        part9_data = sitk.ReadImage(part9_item_path)
        part9_label = sitk.GetArrayFromImage(part9_data).astype(np.uint8)

        roi = 1
        for i in range(1,11):
            final_label[part10_label == i] = roi
            roi += 1
    
        for i in range(1,10):
            final_label[part9_label == i] = roi
            roi += 1
        
        if POST:
            for i in range(final_label.shape[0]):
                final_label[i] = post_process(final_label[i],20)

        sitk_data = sitk.GetImageFromArray(final_label)
        sitk_data.SetSpacing(spacing)
        sitk_data.SetOrigin(origin)
        sitk_data.SetDirection(direction)
        
        save_path = os.path.join(save_folder,part10_item.name)
        sitk.WriteImage(sitk_data, save_path)
