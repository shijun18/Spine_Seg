import os
import json
import glob

from numpy import not_equal

from utils import get_path_with_annotation,get_path_with_annotation_ratio
from utils import get_weight_path

__disease__ = ['Spine']
__net__ = ['unet','unet++','FPN','deeplabv3+']
__encoder_name__ = ['resnet18','resnet34','resnet50','se_resnet50','resnext50_32x4d', 'timm-resnest14d','timm-resnest26d','timm-resnest50d', \
                    'efficientnet-b4', 'efficientnet-b5','efficientnet-b6','efficientnet-b7']

__mode__ = ['cls','seg']


json_path = {
    'Spine':'/staff/shijun/torch_projects/Spine_Seg/converter/static_files/spine.json',
}
    
DISEASE = 'Spine' 
MODE = 'seg'
NET_NAME = 'deeplabv3+'
ENCODER_NAME = 'resnet50'
VERSION = 'v4.3-all'

with open(json_path[DISEASE], 'r') as fp:
    info = json.load(fp)

DEVICE = '1'
# True if use internal pre-trained model
# Must be True when pre-training and inference
PRE_TRAINED = False
# True if use external pre-trained model 
EX_PRE_TRAINED = False
# True if use resume model
CKPT_POINT = False

FOLD_NUM = 5
# [1-FOLD_NUM]
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))

# Arguments for trainer initialization
#--------------------------------- single or multiple
# ROI_NUMBER = None
ROI_NUMBER = [1,2,3,4,5,6,7,8,9,10]
# ROI_NUMBER = [11,12,13,14,15,16,17,18,19]


NUM_CLASSES = info['annotation_num'] + 1  # 2 for binary, more for multiple classes
if ROI_NUMBER is not None:
    if isinstance(ROI_NUMBER,list):
        NUM_CLASSES = len(ROI_NUMBER) + 1
        ROI_NAME = 'Part_{}'.format(str(len(ROI_NUMBER)))
    else:
        NUM_CLASSES = 2
        ROI_NAME = info['annotation_list'][ROI_NUMBER - 1]
else:
    ROI_NAME = 'All'


MEAN = info['mean_std']['mean']
STD = info['mean_std']['std']
#---------------------------------

#--------------------------------- mode and data path setting
#all
PATH_LIST = glob.glob(os.path.join(info['2d_data']['save_path'],'*.hdf5'))

#zero
# PATH_LIST = get_path_with_annotation(info['2d_data']['csv_path'],'path',ROI_NAME)


#half
# PATH_LIST = get_path_with_annotation_ratio(info['2d_data']['csv_path'],'path','T9/T10',ratio=0.5)


#equal
# PATH_LIST = get_path_with_annotation_ratio(info['2d_data']['csv_path'],'path','T9/T10',ratio=1)
#---------------------------------


#--------------------------------- others
INPUT_SHAPE = (512,512)
BATCH_SIZE = 16

CKPT_PATH = './ckpt/{}/{}/{}/{}/fold{}'.format(DISEASE,MODE,VERSION,ROI_NAME,str(CURRENT_FOLD))

WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

INIT_TRAINER = {
  'net_name':NET_NAME,
  'encoder_name':ENCODER_NAME,
  'lr':1e-3, 
  'n_epoch':80,
  'channels':1,
  'num_classes':NUM_CLASSES, 
  'roi_number':ROI_NUMBER,
  'input_shape':INPUT_SHAPE,
  'crop':0,
  'batch_size':BATCH_SIZE,
  'num_workers':2,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'ex_pre_trained':EX_PRE_TRAINED,
  'ckpt_point':CKPT_POINT,
  'weight_path':WEIGHT_PATH,
  'weight_decay': 0.0001,
  'momentum': 0.99,
  'gamma': 0.1,
  'milestones': [20,40],
  'T_max':5,
  'mean':MEAN,
  'std':STD,
  'mode':MODE,
  'topk':20,
  'use_fp16':True #False if the machine you used without tensor core
 }
#---------------------------------

__seg_loss__ = ['DiceLoss','Cross_Entropy','DynamicTopKLoss','TopKLoss']
__cls_loss__ = ['BCEWithLogitsLoss']
# Arguments when perform the trainer 

if MODE == 'cls':
    LOSS_FUN = 'BCEWithLogitsLoss'
elif MODE == 'seg' :
    LOSS_FUN = 'TopKLoss'

SETUP_TRAINER = {
  'output_dir':'./ckpt/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME),
  'log_dir':'./log/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME), 
  'optimizer':'Adam',
  'loss_fun':LOSS_FUN,
  'class_weight':None, #[1,4]
  'lr_scheduler':'MultiStepLR', #'CosineAnnealingLR','MultiStepLR','CosineAnnealingWarmRestarts'
  }
#---------------------------------
TEST_PATH = None