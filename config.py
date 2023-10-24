import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 3035 # random seed,  for reproduction
__C.DATASET = 'SHHB' # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE, Mall, UCSD

if __C.DATASET == 'UCF50':# only for UCF50
	from datasets.UCF50.setting import cfg_data
	__C.VAL_INDEX = cfg_data.VAL_INDEX 

if __C.DATASET == 'GCC':# only for GCC
	from datasets.GCC.setting import cfg_data
	__C.VAL_MODE = cfg_data.VAL_MODE 


__C.NET = 'DDP' # net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet, Res101...

__C.PRE_GCC = False # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = './C3_Framework/weights/VGGPre.pth' # path to model

__C.RESUME = False # contine training
__C.RESUME_PATH = './C3_Framework/weights/nogcc.pth' #

__C.PRE_LORA=False
__C.PRE_LORA_MODEL = './C3_Framework/weights/nogcc.pth' #

__C.GPU_ID = [0] # sigle gpu: [0], [1] ...; multi gpus: [0,1]

# learning rate settings
__C.LR = 1e-5 # learning rate
__C.LR_DECAY = 0.995 # decay rate
__C.LR_DECAY_START = -1 # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1 # decay frequency
__C.MAX_EPOCH = 200

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on

__C.LAMBDA_1 = 1e-4# SANet:0.001 CMTL 0.0001


# print 
__C.PRINT_FREQ = 10

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
			 + '_' + __C.DATASET \
             + '_' + __C.NET \
             + '_' + str(__C.LR)

if __C.DATASET == 'UCF50':
	__C.EXP_NAME += '_' + str(__C.VAL_INDEX)	

if __C.DATASET == 'GCC':
	__C.EXP_NAME += '_' + __C.VAL_MODE	

__C.EXP_PATH = './exp' # the path of logs, checkpoints, and current codes


#------------------------------VAL------------------------
__C.VAL_DENSE_START = 50
__C.VAL_FREQ = 10 # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

#------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1 #  must be 1 for training images with the different sizes
#---------------------------Diffusion---------------------
# __C.image_size=768
# __C.image_size_one=768
# __C.image_size_two=1024
# __C.class_cond=False
# __C.learn_sigma=True
# __C.num_channels=128
# __C.num_res_blocks=2
# __C.channel_mult=''
# __C.in_ch=4
# __C.num_heads=1
# __C.num_head_channels=-1
# __C.num_heads_upsample=-1
# __C.attention_resolutions='16'
# __C.dropout=0.0
# __C.diffusion_steps=1000
# __C.noise_schedule='linear'
# __C.timestep_respacing=''
# __C.use_kl=False
# __C.predict_xstart=False
# __C.rescale_timesteps=False
# __C.rescale_learned_sigmas=False
# __C.use_checkpoint=False
# __C.use_scale_shift_norm=False
# __C.resblock_updown=False
# __C.use_fp16=False
# __C.use_new_attention_order=False
# __C.dpm_solver=False
# __C.schedule_sampler='uniform'



#================================================================================
#================================================================================
#================================================================================  
