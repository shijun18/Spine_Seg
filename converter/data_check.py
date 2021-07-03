import os 
import numpy as np
import json
import pandas as pd
from utils import hdf5_reader



def data_check(input_path,annotation_list):
    slice_num = 0
    csv_info = []
    class_list = []
    for item in os.scandir(input_path):
        csv_item = []
        print(item.name)
        csv_item.append(item.name)
        img = hdf5_reader(item.path,'image')
        lab = hdf5_reader(item.path,'label')
        print(img.shape)
        slice_num += img.shape[0]
        csv_item.append(img.shape[0])
        csv_item.append(np.max(img))
        csv_item.append(np.min(img))
        print(np.max(img),np.min(img))
        print(np.unique(lab))
        class_list.extend(list(np.unique(lab))[1:])
        csv_info.append(csv_item)

    col = ['id','slices_num','max','min']
    csv_file = pd.DataFrame(columns=col, data=csv_info)
    csv_file.to_csv('./data_check.csv', index=False)

    print('total slice: %d'%slice_num)

    for i in range(len(annotation_list)):
        print('%s : %d'%(annotation_list[i],class_list.count(i+1)))

def cal_mean_std(data_path):
    image = []
    for item in os.scandir(data_path):
        img = hdf5_reader(item.path,'image').flatten()
        image.extend(img)
        
    print('mean:%.3f' % np.mean(image))
    print('std:%.3f' % np.std(image))
        

if __name__ == "__main__":
    json_file = './static_files/spine.json'
    with open(json_file, 'r') as fp:
        info = json.load(fp)
    data_check(info['npy_path'],info['annotation_list'])
    cal_mean_std(info['npy_path'])