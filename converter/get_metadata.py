import os,glob
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import json


def metadata_reader(data_path):

    info = []
    data = sitk.ReadImage(data_path)
    # print(data)
    size = list(data.GetSize()[:2])
    z_size = data.GetSize()[-1]
    thick_ness = data.GetSpacing()[-1]
    pixel_spacing = list(data.GetSpacing()[:2])
    info.append(size)
    info.append(z_size)
    info.append(thick_ness)
    info.append(pixel_spacing)
    return info

# Different samples are saved in different folder
def get_metadata(input_path, save_path):
    image_path = os.path.join(input_path,'MR')
    info = []
    for item in tqdm(os.scandir(image_path)):
        info_item = [item.name.split('.')[0]]
        info_item.extend(metadata_reader(item.path))
        info.append(info_item)
    col = ['id', 'size', 'num', 'thickness', 'pixel_spacing']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(save_path, index=False)



if __name__ == "__main__":

    json_file = './static_files/spine.json'
    with open(json_file, 'r') as fp:
        info = json.load(fp)
    get_metadata(info['nii_path'], info['metadata_path'])
