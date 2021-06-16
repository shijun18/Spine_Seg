import sys
import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

# num_classes = 10 
num_classes = 9

# result_list = ['./result/Spine/origin/v4.10-all/Part_10/fold1',\
#                './result/Spine/origin/v4.10-all/Part_10/fold2',\
#                './result/Spine/origin/v4.10-all/Part_10/fold3',
#                './result/Spine/origin/v4.10-all/Part_10/fold4',
#                './result/Spine/origin/v4.10-all/Part_10/fold5']

# save_folder = './result/Spine/v4.10-all/Part_10/fusion/'


result_list = ['./result/Spine/origin/v4.10-all/Part_9/fold1',\
               './result/Spine/origin/v4.10-all/Part_9/fold2',\
               './result/Spine/origin/v4.10-all/Part_9/fold3',
               './result/Spine/origin/v4.10-all/Part_9/fold4',
               './result/Spine/origin/v4.10-all/Part_9/fold5']

save_folder = './result/Spine/v4.10-all/Part_9/fusion/'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for item in tqdm(os.listdir(result_list[0])):
    img_list = [os.path.join(case, item) for case in result_list]
    data = sitk.ReadImage(img_list[0])
    label = sitk.GetArrayFromImage(data).astype(np.uint8)
    
    spacing = data.GetSpacing()
    origin = data.GetOrigin()
    direction = data.GetDirection()

    final_label = np.zeros_like(label,dtype=np.uint8)
    for z in range(num_classes):
        tmp_roi = np.zeros_like(final_label,dtype=np.uint8)
        for img_path in img_list:
            data = sitk.ReadImage(img_path)
            tmp_label = sitk.GetArrayFromImage(data).astype(np.uint8)
            tmp_roi += (tmp_label==z+1).astype(np.uint8)
        final_label[tmp_roi >= len(result_list)//2] = z+1
    
    sitk_data = sitk.GetImageFromArray(final_label)
    sitk_data.SetSpacing(spacing)
    sitk_data.SetOrigin(origin)
    sitk_data.SetDirection(direction)
    
    save_path = os.path.join(save_folder,item)
    sitk.WriteImage(sitk_data, save_path)
