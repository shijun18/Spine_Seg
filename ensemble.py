import sys
import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm



root_path = './result/Spine/origin/v4.3-all/'

for part in [9,10]:
    num_classes = part

    result_list = [root_path + f'Part_{part}/' + f'fold{case}' for case in range(1,6)]
    save_folder = './result/Spine/v4.3-all/Part_{}/fusion/'.format(part)

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
            final_label[tmp_roi > len(result_list)//2] = z+1
        
        sitk_data = sitk.GetImageFromArray(final_label)
        sitk_data.SetSpacing(spacing)
        sitk_data.SetOrigin(origin)
        sitk_data.SetDirection(direction)
        
        save_path = os.path.join(save_folder,item)
        sitk.WriteImage(sitk_data, save_path)
