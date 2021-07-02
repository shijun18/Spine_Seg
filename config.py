import os
import json
import glob

from numpy import not_equal

from utils import get_path_with_annotation,get_path_with_annotation_ratio
from utils import get_weight_path

__disease__ = ['Spine']
__net__ = ['unet','unet++','FPN','deeplabv3+','swin_trans_unet']
__encoder_name__ = [None,'resnet18','resnet34','resnet50','se_resnet50','resnext50_32x4d', 'timm-resnest14d','timm-resnest26d','timm-resnest50d', \
                    'efficientnet-b4', 'efficientnet-b5','efficientnet-b6','efficientnet-b7']

__mode__ = ['cls','seg','mtl']


json_path = {
    'Spine':'/staff/shijun/torch_projects/Spine_Seg/converter/static_files/spine.json',
}
    
DISEASE = 'Spine' 
MODE = 'seg'
NET_NAME = 'deeplabv3+'
ENCODER_NAME = 'resnet50'
VERSION = 'v4.3-balance'

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
# ROI_NUMBER = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18]# or [1-N]
# ROI_NUMBER = [1,2,3,4,5,6,7,8,11,12,13,14,15,16,17]# or [1-N]
ROI_NUMBER = [1,2,3,4,5,6,7,8,9,10]
# ROI_NUMBER = [11,12,13,14,15,16,17,18,19]
# ROI_NUMBER = [9,10,18,19]
# ROI_NUMBER = [10,19]

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

# all
if 'all' in VERSION:
    #all
    PATH_LIST = glob.glob(os.path.join(info['2d_data']['save_path'],'*.hdf5'))
    PATH_LIST += glob.glob(os.path.join(info['2d_data']['test_path'],'*.hdf5'))

#balance

elif 'balance' in VERSION:
    if ROI_NAME == 'Part_10':
        PATH_LIST = []
        alpha_list = glob.glob(os.path.join(info['2d_data']['save_path'],'*.hdf5'))
        alpha_list += glob.glob(os.path.join(info['2d_data']['test_path'],'*.hdf5'))

        beta_list = get_path_with_annotation(info['2d_data']['csv_path'],'path','T9')
        beta_list += get_path_with_annotation(info['2d_data']['test_csv_path'],'path','T9')

        alpha_list = [case for case in alpha_list if case not in beta_list]
        PATH_LIST.append(alpha_list)
        PATH_LIST.append(beta_list)
    
    elif ROI_NAME == 'Part_9':
        PATH_LIST = []
        alpha_list = glob.glob(os.path.join(info['2d_data']['save_path'],'*.hdf5'))
        alpha_list += glob.glob(os.path.join(info['2d_data']['test_path'],'*.hdf5'))

        beta_list = get_path_with_annotation(info['2d_data']['csv_path'],'path','T9/T10')
        beta_list += get_path_with_annotation(info['2d_data']['test_csv_path'],'path','T9/T10')

        alpha_list = [case for case in alpha_list if case not in beta_list]
        PATH_LIST.append(alpha_list)
        PATH_LIST.append(beta_list)

#---------------------------------


#--------------------------------- others
INPUT_SHAPE = (512,512)
BATCH_SIZE = 32

CKPT_PATH = './ckpt/{}/{}/{}/{}/fold{}'.format(DISEASE,MODE,VERSION,ROI_NAME,str(CURRENT_FOLD))

WEIGHT_PATH = get_weight_path(CKPT_PATH)
# print(WEIGHT_PATH)

INIT_TRAINER = {
  'net_name':NET_NAME,
  'encoder_name':ENCODER_NAME,
  'lr':1e-3, 
  'n_epoch':100,
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
  'milestones': [20,40,60],
  'T_max':5,
  'mean':MEAN,
  'std':STD,
  'mode':MODE,
  'topk':20,
  'freeze':None,
  'use_fp16':True, #False if the machine you used without tensor core
  'statistic_threshold':True,
  'prefetch':False
 }
#---------------------------------

__seg_loss__ = ['TopKLoss','DiceLoss','Cross_Entropy','BCELoss','TopKBCELoss','DynamicTopKLoss','CEPlusDice','TopkCEPlusDice']
__cls_loss__ = ['BCEWithLogitsLoss']
__mtl_loss__ = ['BCEPlusDice']
# Arguments when perform the trainer 

if MODE == 'cls':
    LOSS_FUN = 'BCEWithLogitsLoss'
elif MODE == 'seg' :
    LOSS_FUN = 'TopKLoss'
    # LOSS_FUN = 'DiceLoss'
    # LOSS_FUN = 'Cross_Entropy'
    # LOSS_FUN = 'BCELoss'
    # LOSS_FUN = 'TopKBCELoss'
    # LOSS_FUN = 'TopkCEPlusDice'
else:
    LOSS_FUN = 'BCEPlusDice'

SETUP_TRAINER = {
  'output_dir':'./ckpt/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME),
  'log_dir':'./log/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME), 
  'csv_path':'./csv_file/{}/{}/{}'.format(DISEASE,VERSION,ROI_NAME), 
  'optimizer':'Adam',
  'loss_fun':LOSS_FUN,
  'class_weight':None, #[1,4]
  'lr_scheduler':'MultiStepLR', #'CosineAnnealingLR','MultiStepLR'
  'balance':True if 'balance' in VERSION else False
  }
#---------------------------------
TEST_PATH = None