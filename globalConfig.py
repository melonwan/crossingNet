import sys, os

if len(sys.argv) < 2:
    raise ValueError('should input the dataset')

dataset = sys.argv[1]
dataset = dataset.upper()
print 'globalConfig.dataset', dataset

# path to the corresponding dataset
msra_base_path = '/home/wanc/data/hand/msra15'
nyu_base_path = '/home/wanc/data/hand/nyu/dataset/'
icvl_base_path = '/scratch_net/unclemax/wanc/data/hand/icl/'
cache_base_path = './cache/data/'

# path to save the network parameter, intermediate outputs and test result
model_dir = './cache/model/' 

# path to the pretrained_model of depthGAN and poseVAE
gan_pretrain_path = os.path.join(model_dir, 'depth_gan', '%s_dummy/params/-1'%dataset)
vae_pretrain_path = os.path.join(model_dir, 'pose_vae', '%s_dummy/params/-1'%dataset)
