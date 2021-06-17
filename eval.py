import os
import numpy as np
from utils import get_weight_path
from metrics import multi_dice
from trainer import SemanticSeg

ANNOTATION_LIST = ["S","L5","L4","L3","L2","L1","T12","T11","T10","T9","L5/S","L4/L5","L3/L4","L2/L3","L1/L2","T12/L1","T11/T12","T10/T11","T9/T10"]

DISEASE = 'Spine' 
MODE = 'seg'
NET_NAME = 'deeplabv3+'
ENCODER_NAME = 'efficientnet-b5'
VERSION = 'v4.10-all'

DEVICE = '1'
CURRENT_FOLD = 1


# Arguments for trainer initialization
#--------------------------------- single or multiple
# ROI_NUMBER = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18]# or [1-N]
# ROI_NUMBER = [1,2,3,4,5,6,7,8,11,12,13,14,15,16,17]# or [1-N]
ROI_NUMBER = [1,2,3,4,5,6,7,8,9,10]
# ROI_NUMBER = None
# ROI_NUMBER = [9,10,18,19]
# ROI_NUMBER = [10,19]

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


INPUT_SHAPE = (512,512)
BATCH_SIZE = 16

CKPT_PATH = './ckpt/{}/{}/{}/{}/fold{}'.format(DISEASE,MODE,VERSION,ROI_NAME,str(CURRENT_FOLD))

WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

INIT_TRAINER = {
  'net_name':NET_NAME,
  'encoder_name':ENCODER_NAME,
  'channels':1,
  'num_classes':NUM_CLASSES, 
  'roi_number':ROI_NUMBER,
  'input_shape':INPUT_SHAPE,
  'crop':0,
  'batch_size':16,
  'num_workers':2,
  'device':DEVICE,
  'pre_trained':True,
  'weight_path':WEIGHT_PATH,
  'mean':239,
  'std':257,
  'use_fp16':True, #False if the machine you used without tensor core
  'mode':MODE
 }

segnetwork = SemanticSeg(**INIT_TRAINER)

data_path = '/staff/shijun/dataset/Med_Seg/Spine/2d_test_data'
sample_list = ['Case202']

total_dice = []

for sample in sample_list:
    test_path = [case.path for case in os.scandir(data_path) if case.name.split('_')[0] == sample ]
    test_path.sort(key=lambda x:eval(x.split('_')[-1].split('.')[0]))
    seg_result,_ = segnetwork.test(test_path)

    pred = np.concatenate(seg_result['pred'],axis=0)
    true = np.concatenate(seg_result['true'],axis=0)

    category_dice, avg_dice = multi_dice(true,pred,NUM_CLASSES - 1)
    total_dice.append(category_dice)
    print('category dice:',category_dice)
    print('avg dice: %s'% avg_dice)


total_dice = np.stack(total_dice,axis=0)
total_category_dice = np.mean(total_dice,axis=0)
total_avg_dice = np.mean(total_category_dice)
print('total category dice:',total_category_dice)
print('totalavg dice: %s'% total_avg_dice)