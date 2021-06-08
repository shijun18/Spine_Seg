import os,glob
import pandas as pd
import h5py
import numpy as np
import torch
import random
from matplotlib import pyplot as plt 


def draw_contour(image,pred_label,true_label,n):
    #image：原图  label:类别大小以数字区分  n：为器官数量
    colors = ['red','green','blue','orange','cyan','purple','palegreen','pink','brown','olive','tomato','crimson','indigo',\
            'turquoise','seagreen','lightskyblue','lime','sienna','fuchsia'] 
    plt.figure(figsize=(8,16))
    plt.axis('off')
    final_labels =  np.zeros(pred_label.shape + (n,), dtype=np.uint8)
    plt.subplot(121)
    plt.imshow(image,'gray')
    for z in range(n):
        final_labels[pred_label==z+1,z]=1
    for i in range(n):
        if np.sum(final_labels[:,:,i]):
            plt.contour(final_labels[:,:,i],colors = colors[i])

    final_labels =  np.zeros(true_label.shape + (n,), dtype=np.uint8)
    plt.subplot(122)
    plt.imshow(image,'gray')
    for z in range(n):
        final_labels[true_label==z+1,z]=1
    for i in range(n):
        if np.sum(final_labels[:,:,i]):
            plt.contour(final_labels[:,:,i],colors = colors[i])

    plt.show()
    plt.close()


def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


def get_path_with_annotation(input_path,path_col,tag_col):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    final_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            final_list.append(path)
    
    return final_list


def get_path_with_annotation_ratio(input_path,path_col,tag_col,ratio=0.5):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    with_list = []
    without_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            with_list.append(path)
        else:
            without_list.append(path)
    if int(len(with_list)/ratio) < len(without_list):
        random.shuffle(without_list)
        without_list = without_list[:int(len(with_list)/ratio)]    
    return with_list + without_list


def count_params_and_macs(net,input_shape):
    
    from thop import profile
    input = torch.randn(input_shape)
    input = input.cuda()
    macs, params = profile(net, inputs=(input, ))
    print('%.3f GFLOPs' %(macs/10e9))
    print('%.3f M' % (params/10e6))



def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        return None
    
    

def remove_weight_path(ckpt_path,retain=10):

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