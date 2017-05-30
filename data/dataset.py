'''
import the sequence and pre-process it
'''
import globalConfig
import progressbar as pb, numpy as np
from numpy.matlib import repmat
from numpy.linalg import svd, det
from depth import DepthMap
from util import Frame, Camera
from geometry import Quaternion, Matrix33
import cPickle, os
import time
import scipy.io as sio
import globalConfig

# for pickle module
import sys, util
sys.modules['util'] = util

class Dataset(object):
    '''
    base type to load dataset and as interface to the training
    '''
    def __init__(self):
        self.dataset= globalConfig.dataset
        print('initialized')

        if self.dataset == 'NYU':
            self.refPtIdx = [31,32,33,34,35]
            self.skel_num = 36
            self.centerPtIdx = 32 
        elif self.dataset == 'MSRA':
            self.refPtIdx = [1,5,9,13,17]
            self.skel_num = 21
            self.centerPtIdx = 0
        elif self.dataset == 'ICVL':
            self.refPtIdx = [1,4,7,10,13]
            self.skel_num = 16 
            self.centerPtIdx = 0

        self.cache_base_path = globalConfig.cache_base_path
        self.msra_base_path = globalConfig.msra_base_path
        self.msra_pose_list = '1  2  3  4  5  6  7  8  9  I  IP  L  MP  RP  T  TIP  Y'.split()
        self.nyu_base_path = globalConfig.nyu_base_path
        self.nyu_frm_perfile = 20000 # the maximum number of frame to store in each file
        self.icvl_base_path = globalConfig.icvl_base_path

    def loadMSRA(self, seqName, mode='train', replace=False, tApp=False):
        '''seqName: P0 - P8
           mode: if train, only save the cropped image
           replace: replace the previous cache file if exists
           tApp: append to previous loaded file if True
        '''
        if not hasattr(self, 'frmList'):
            self.frmList =[]
        if not tApp:
            self.frmList = []

        pickleCachePath = '{}/msra_{}.pkl'.format(self.cache_base_path, seqName)
        if os.path.isfile(pickleCachePath) and not replace:
            print 'direct load from the cache'
            t1 = time.time()
            f = open(pickleCachePath, 'rb')
            (self.frmList) += cPickle.load(f)
            t1 = time.time() - t1
            print 'loaded with {}s'.format(t1)
            return self.frmList

        Camera.setCamera('INTEL')
        pbar = pb.ProgressBar(maxval = 500*len(self.msra_pose_list), widgets = ['Loading MSRA | ', pb.Percentage(), pb.Bar()])
        pbar.start()
        pbIdx = 0

        seqPath = '/'.join([self.msra_base_path, seqName])
        for pose_name in self.msra_pose_list:
            curPath = '/'.join([seqPath, pose_name, 'joint.txt'])
            f = open(curPath, 'r')
            frmNum = int(f.readline()[:-1])
            for frmIdx in range(frmNum):
                frmPath = '/'.join([seqPath, pose_name, '%06i_depth.bin'%(frmIdx)])
                dm = DepthMap('MSRA', frmPath)
                skel = f.readline().split()
                skel = np.asarray([float(pt) for pt in skel])
                def cvtMSRA_skel(init_skel):
                    skel = init_skel.copy()
                    for i in range(len(skel)):
                        if i%3 == 2:
                            skel[i] *= -1.0
                    return skel
                skel = cvtMSRA_skel(skel)
                self.frmList.append(Frame(dm, skel))
                if mode is 'train':
                    self.frmList[-1].saveOnlyForTrain()
                pbar.update(pbIdx)
                pbIdx += 1
        pbar.finish()

        if not os.path.exists(self.cache_base_path):
            os.makedirs(self.cache_base_path)
        f = open(pickleCachePath, 'wb')
        cPickle.dump((self.frmList), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print 'loaded with {} frames'.format(len(self.frmList))

    def loadICVLTest(self):
        self.frmList = [] 

        pickleCachePath = '{}/icvl_test.pkl'.format(self.cache_base_path)
        if os.path.isfile(pickleCachePath):
            print 'direct load from the cache'
            t1 = time.time()
            f = open(pickleCachePath, 'rb')
            (self.frmList) += cPickle.load(f)
            t1 = time.time() - t1
            print 'loaded with {}s'.format(t1)
            return

        label_path = os.path.join(self.icvl_base_path,
                                  'Testing/labels.txt') 
        labels = [line for line in open(label_path)]
        print 'ICVL testing: %d sequences in total'%(len(labels))

        Camera.setCamera('INTEL')
        pbar = pb.ProgressBar(maxval = len(labels), widgets = ['Loading ICVL | ', pb.Percentage(), pb.Bar()])
        pbar.start()
        pbIdx = 0
        for label in labels:
            label_cache = label.split(' ')
            frmPath = os.path.join(self.icvl_base_path, 
                                  'Testing/Depth',
                                  label_cache[0])
            label_cache = label_cache[0:49]
            skel = np.asarray([float(j) for j in label_cache[1:]])
            skel.shape = (-1,3)
            for idx, pt in enumerate(skel):
                skel[idx] = Camera.to3D(pt)
            skel.shape = (-1)
            dm = DepthMap('ICVL', frmPath)
            dm.dmData[dm.dmData >= 500] = 32001
            self.frmList.append(Frame(dm, skel))
            pbar.update(pbIdx)
            pbIdx += 1
        pbar.finish()

        if not os.path.exists(self.cache_base_path):
            os.makedirs(self.cache_base_path)
        f = open(pickleCachePath, 'wb')
        cPickle.dump((self.frmList), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print 'loaded with {} frames'.format(len(self.frmList))

    def loadICVL(self, seqName = '2014', tApp=False, tReplace=False):
        '''seqName: corresponding folder names in the icvl dataset 
           mode: if train, only save the cropped image
           replace: replace the previous cache file if exists
           tApp: append to previous loaded file if True
        '''
        if not hasattr(self, 'frmList'):
            self.frmList =[]
        if not tApp:
            self.frmList = []

        pickleCachePath = '{}/icvl_{}.pkl'.format(self.cache_base_path, seqName)
        if os.path.isfile(pickleCachePath) and not tReplace:
            print 'direct load from the cache'
            t1 = time.time()
            f = open(pickleCachePath, 'rb')
            (self.frmList) += cPickle.load(f)
            t1 = time.time() - t1
            print 'loaded with {}s'.format(t1)
            return

        label_path = os.path.join(self.icvl_base_path,
                                  'Training/labels.txt') 
        labels = [line for line in open(label_path) if line.startswith(seqName)]
        print '%s: %d sequences in total'%(seqName, len(labels))

        Camera.setCamera('INTEL')
        pbar = pb.ProgressBar(maxval = len(labels), widgets = ['Loading ICVL | ', pb.Percentage(), pb.Bar()])
        pbar.start()
        pbIdx = 0
        for label in labels:
            label_cache = label.split(' ')
            frmPath = os.path.join(self.icvl_base_path, 
                                  'Training/Depth',
                                  label_cache[0])
            skel = np.asarray([float(j) for j in label_cache[1:]])
            skel.shape = (-1,3)
            for idx, pt in enumerate(skel):
                skel[idx] = Camera.to3D(pt)
            skel.shape = (-1)
            dm = DepthMap('ICVL', frmPath)
            self.frmList.append(Frame(dm, skel))
            self.frmList[-1].saveOnlyForTrain()
            pbar.update(pbIdx)
            pbIdx += 1
        pbar.finish()

        if not os.path.exists(self.cache_base_path):
            os.makedirs(self.cache_base_path)
        f = open(pickleCachePath, 'wb')
        cPickle.dump((self.frmList), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print 'loaded with {} frames'.format(len(self.frmList))

    def loadNYU(self, frmStartNum, cameraIdx = 1, tFlag = 'train', tApp =\
                False, isReplace=False):
        '''frmStartNum: starting frame index
           cameraIdx: [1,3]
           tFlag: save only the cropped image if is 'train'
           tApp: append to the previously loaded file if True
        '''
        Camera.setCamera('KINECT')
        if cameraIdx not in [1]:
            raise ValueError('invalid cameraIdx, current only support view from 1')

        if tFlag not in ['train', 'test']:
            raise ValueError('invalid tFlag, can be only train or test')

        # load the annotation file
        matPath = '{}/{}/joint_data.mat'.format(self.nyu_base_path, tFlag)
        joint = sio.loadmat(matPath)
        joint_xyz = joint['joint_xyz'][cameraIdx-1]
        joint_uvd = joint['joint_uvd'][cameraIdx-1]
        matPath = './data/center_uvd_{}.mat'.format(tFlag)
        center = sio.loadmat(matPath)
        center = center['center_uvd']

        # determine the start and end frame
        if frmStartNum >= len(joint_xyz):
            raise ValueError('invalid start frame, shoud be lower than {}'.format(len(joint_xyz)))

        fileIdx = int(frmStartNum / self.nyu_frm_perfile)
        frmStartNum = fileIdx*self.nyu_frm_perfile
        if tFlag == 'train':
            frmEndNum = min(frmStartNum+self.nyu_frm_perfile, len(joint_xyz))
        elif tFlag == 'test':
            frmEndNum = len(joint_xyz)
        print 'frmStartNum={}, frmEndNum={}, fileIdx={}'.format(frmStartNum,
                                                                frmEndNum,
                                                                fileIdx)

        pickleCachePath = '{}/nyu_{}_{}_{}.pkl'.format(self.cache_base_path,
                                                    tFlag, cameraIdx, fileIdx)
        if not hasattr(self, 'frmList'):
            self.frmList =[]
        if not tApp:
            self.frmList = []

        if os.path.isfile(pickleCachePath) and isReplace == False:
            print 'direct load from the cache'
            print 'cache dir ={}'.format(pickleCachePath)
            t1 = time.time()
            f = open(pickleCachePath, 'rb')
            self.frmList += cPickle.load(f)
            t1 = time.time() - t1
            print 'loaded with {}s'.format(t1)
            return

        pbar = pb.ProgressBar(maxval = frmEndNum-frmStartNum, widgets = ['Loading NYU | ', pb.Percentage(), pb.Bar()])
        pbar.start()
        pbIdx = 0

        for frmIdx in range(frmStartNum, frmEndNum):
            frmPath = '{}/{}/depth_{}_{:07d}.png'.format(self.nyu_base_path, tFlag,
                                                       cameraIdx, frmIdx+1)
            dm = DepthMap('NYU', frmPath)
            skel = joint_xyz[frmIdx]
            skel = np.reshape(skel, (-1))
            com_uvd = center[frmIdx]
            self.frmList.append(Frame(dm, skel, com_uvd))
            self.frmList[-1].saveOnlyForTrain()
            pbar.update(pbIdx)
            pbIdx += 1
        pbar.finish()

        if not os.path.exists(self.cache_base_path):
            os.makedirs(self.cache_base_path)
        f = open(pickleCachePath, 'wb')
        cPickle.dump((self.frmList), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print 'loaded with {} frames'.format(len(self.frmList))

    '''
    interface to neural network, used for training
    '''
    def normTranslation(self, origin_pt_idx = None):
        if origin_pt_idx is None:
            origin_pt_idx = self.centerPtIdx
        if not hasattr(self, 'frmList') or self.frmList == None:
            return

        if not hasattr(self.frmList[0], 'norm_skel'):
            return
    
        vec_dim = 3*self.skel_num
        for frm in self.frmList:
            frm.origin =\
                frm.norm_skel[origin_pt_idx*3:origin_pt_idx*3+3].copy()
            origin = repmat(frm.origin, 1, self.skel_num)
            origin.shape = (vec_dim,)
            frm.norm_skel -= origin 

    def normRotation(self, tmpSkel = None, refPtIdx = None):
        '''tmpSkel: normalize every palm pose to the tmpSkel pose (template skeleton)
           refPtIdx: indexes of joints on the palm
        '''
        if tmpSkel is None:
            tmpSkel = self.frmList[0].norm_skel
        
        if refPtIdx is None:
            refPtIdx = self.refPtIdx
        refIdx = []
        for idx in refPtIdx:
            refIdx += [idx*3, idx*3+1, idx*3+2]
        
        keep_list = set(range(3*self.skel_num)).\
            difference(set(refIdx+range(self.centerPtIdx, self.centerPtIdx+3)))
        keep_list = list(keep_list)

        temp = tmpSkel[refIdx].copy()
        temp.shape = (-1,3)

        for frm in self.frmList:
           model = frm.norm_skel[refIdx] 
           model.shape = (-1,3)
           
           R = np.zeros((3,3), np.float32)
           for vt, vm in zip(temp, model):
               R = R + np.dot(vm.reshape(3,1), vt.reshape(1,3))
            
           U,s,V = svd(R, full_matrices=True) 
           R = np.dot(V.transpose(), U.transpose())
           frm.quad = Quaternion(R)
           frm.norm_skel.shape = (-1,3)
           frm.norm_skel = np.dot(R,frm.norm_skel.transpose())
           frm.norm_skel = frm.norm_skel.flatten('F')
           # frm.norm_skel = frm.norm_skel[keep_list]
    
    def skelNum(self):
        if self.frmList is None or len(self.frmList) == 0:
            raise ValueError('frameList is empty')
        return len(self.frmList[0].norm_skel)

    def frmToNp(self):
        '''
        prepare the training samples for training the neural network
        normalize the input and output data
        '''
        if self.frmList is None or len(self.frmList) == 0:
            raise ValueError('frameList is empty')

        frmNum = len(self.frmList)
        jntNum = len(self.frmList[0].norm_skel)

        # normalized
        self.x_norm = np.zeros((frmNum, 1, DepthMap.size2[1], DepthMap.size2[0]), np.float32)
        self.y_norm = np.zeros((frmNum, jntNum), np.float32)
        self.pose_orig = np.zeros((frmNum, 3), np.float32)
        self.pose_trans = np.zeros((frmNum, 3, 3), np.float32)
        self.pose_com = np.zeros((frmNum, 3), np.float32)
        
        for i, frm in enumerate(self.frmList):
            if np.any(np.isnan(frm.norm_dm)):
                self.x_norm[i] = self.x_norm[max(i-1,0)]
                self.y_norm[i] = self.y_norm[max(i-1,0)]
                self.pose_orig[i] = self.pose_orig[max(i-1,0)] 
                self.pose_trans[i] = self.pose_trans[max(i-1,0)]
                self.pose_com[i] = self.pose_com[max(i-1,0)]
                continue
            if np.any(np.isnan(frm.norm_skel)):
                self.x_norm[i] = self.x_norm[max(i-1,0)]
                self.y_norm[i] = self.y_norm[max(i-1,0)]
                self.pose_orig[i] = self.pose_orig[max(i-1,0)] 
                self.pose_trans[i] = self.pose_trans[max(i-1,0)]
                self.pose_com[i] = self.pose_com[max(i-1,0)]
                continue

            # reverse the background color to background
            frm.norm_dm[frm.norm_dm==frm.norm_dm.min()] = 0.5
            frm.norm_dm[frm.norm_dm != 0.5] -= frm.norm_dm.min()+0.5
            frm.norm_dm *= np.float32(2)

            self.x_norm[i,0] = frm.norm_dm
            self.y_norm[i] = frm.norm_skel
            if hasattr(frm, 'origin'):
                self.pose_orig[i] = frm.origin
            self.pose_trans[i] = frm.trans
            self.pose_com[i] = frm.com3D

        print 'x_norm range: {} to {}'.format(self.x_norm.min(),
                                              self.x_norm.max())
        return self.x_norm, self.y_norm

if __name__ == '__main__':
    dataset = globalConfig.dataset
    if dataset == 'NYU':
        ds = Dataset()
        for camera_idx in {1}:
            for start_frm in range(0, 75000, 20000):
                ds.loadNYU(start_frm, camera_idx, 'train', isReplace=False)
            for start_frm in range(0, 8000, 20000):
                ds.loadNYU(start_frm, camera_idx, 'test', isReplace=False)

    if dataset == 'MSRA':
        ds = Dataset()
        for p in range(9):
            ds.loadMSRA('P%d'%p, mode='train', replace=True)

    if dataset == 'ICVL':
        ds = Dataset()
        seqNames = '22-5 45 67-5 90 112-5 135 157-5 -22-5 -45 -67-5 -90 -112-5 -157-5 -180'
        for seqName in seqNames.split(' '):
            ds.loadICVL(seqName)

