import os
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast
from skimage.transform import resize
from tqdm import tqdm

from utils import get_weight_path,nii_reader,save_as_nii
from trainer import SemanticSeg


version_list = ['v4.3-balance','v4.3-all','v4.10-balance']
encorder_name = ['resnet50','resnet50','efficientnet-b5']

ANNOTATION_LIST = ["S","L5","L4","L3","L2","L1","T12","T11","T10","T9","L5/S","L4/L5","L3/L4","L2/L3","L1/L2","T12/L1","T11/T12","T10/T11","T9/T10"]

DISEASE = 'Spine' 
MODE = 'seg'
NET_NAME = 'deeplabv3+'

# ENCODER_NAME = 'efficientnet-b5'
ENCODER_NAME = 'resnet50'
# VERSION = 'v4.3-all'
VERSION = 'v4.3-balance'
# VERSION = 'v4.10-balance'

DEVICE = '1'
CURRENT_FOLD = 1


# Arguments for trainer initialization
#--------------------------------- single or multiple
ROI_NUMBER = None
# ROI_NUMBER = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18]# or [1-N]
# ROI_NUMBER = [1,2,3,4,5,6,7,8,11,12,13,14,15,16,17]# or [1-N]
# ROI_NUMBER = [1,2,3,4,5,6,7,8,9,10]
# ROI_NUMBER = [11,12,13,14,15,16,17,18,19]
# ROI_NUMBER = [9,10,18,19]
# ROI_NUMBER = [10,19]

for VERSION,ENCODER_NAME in zip(version_list,encorder_name):
    for ROI_NUMBER in [[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19]]:

        NUM_CLASSES = 20 if ROI_NUMBER is None else len(ROI_NUMBER) + 1
        if ROI_NUMBER is not None:
            if isinstance(ROI_NUMBER,list):
                NUM_CLASSES = len(ROI_NUMBER) + 1
                ROI_NAME = 'Part_{}'.format(str(len(ROI_NUMBER)))
            else:
                NUM_CLASSES = 2
                ROI_NAME = ANNOTATION_LIST[ROI_NUMBER - 1]
        else:
            ROI_NAME = 'All'


        INPUT_SHAPE= (512,512)


        for CURRENT_FOLD in range(1,4):
            print("fold %d start!"%(CURRENT_FOLD))
            CKPT_PATH = './ckpt/{}/{}/{}/{}/fold{}'.format(DISEASE,MODE,VERSION,ROI_NAME,str(CURRENT_FOLD))

            WEIGHT_PATH = get_weight_path(CKPT_PATH)
            print(WEIGHT_PATH)

            INIT_TRAINER = {
            'net_name':NET_NAME,
            'encoder_name':ENCODER_NAME,
            'channels':1,
            'num_classes':NUM_CLASSES, 
            'input_shape':INPUT_SHAPE,
            'crop':0,
            'device':DEVICE,
            'pre_trained':True,
            'weight_path':WEIGHT_PATH,
            'mean':239,
            'std':257,
            'use_fp16':True, #False if the machine you used without tensor core
            'mode':MODE
            }


            os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE
            segnetwork = SemanticSeg(**INIT_TRAINER)
            net = segnetwork.net
            net = net.cuda()

            test_path = '/staff/shijun/dataset/Spine/test2/MR'
            save_path = './post_result/{}/origin/{}/{}/fold{}'.format(DISEASE,VERSION,ROI_NAME,str(CURRENT_FOLD))
            sample_list = [case.path for case in os.scandir(test_path)]

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            net.eval()
            with torch.no_grad():
                for sample in tqdm(sample_list):
                    base_name = os.path.basename(sample).lower()
                    nii_path = os.path.join(save_path,'seg_' + base_name)
                    info,raw_image = nii_reader(sample)

                    # preprocess
                    ## normalize 
                    mean, std, eps = 239, 257, 1e-4
                    image = raw_image - mean
                    image = image / (std + eps)

                    ## resize
                    dim = (image.shape[0],) + INPUT_SHAPE
                    image = resize(image, dim, anti_aliasing=True) #D,H,W

                    ## expand dims
                    image = np.expand_dims(image, axis=0) #1,D,H,W
                    image = np.transpose(image,(1,0,2,3)) #D,1,H,W

                    image = torch.from_numpy(image).cuda()
                    with autocast(True):
                        output = net(image)

                    seg_output = output[0].float()
                    seg_output = torch.softmax(seg_output, dim=1)
                    seg_output = torch.argmax(seg_output,1).detach().cpu().numpy() #D,H,W
                    
                    #post process
                    label = seg_output.astype(np.float32) 
                    ## resize
                    final_label = np.zeros_like(raw_image,dtype=np.uint8)
                    for i in range(1,NUM_CLASSES):
                        roi = resize((label == i).astype(np.float32),raw_image.shape,mode='constant')
                        final_label[roi >= 0.5] = i
                    
                    print(np.unique(final_label))

                    save_as_nii(final_label,nii_path,info)

            print("fold %d end!"%(CURRENT_FOLD))
                


