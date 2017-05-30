# used as the forward path to link the latent space of PoseVAE and DepthGAN
import sys, os, cv2
from data.geometry import Quaternion, Matrix33
from data.util import Camera

from depthGAN import DepthGAN
from poseVAE import PoseVAE

from data.dataset import Dataset
import numpy as np
from numpy.random import RandomState
import time
from numpy.matlib import repmat
from lasagne.layers import batch_norm
import lasagne
import theano, theano.tensor as T
from data.stream import MultiDataStream
import data.util
import globalConfig
if globalConfig.dataset == 'ICVL':
    colorPlatte = data.util.iclColorIdx
    bones = data.util.icvlBones
elif globalConfig.dataset == 'NYU':
    colorPlatte = data.util.nyuColorIdx
    bones = data.util.nyuBones
elif globalConfig.dataset == 'MSRA':
    colorPlatte = data.util.msraColorIdx
    bones = data.util.msraBones
else:
    raise NotImplementedError('unkonwn dataset %s'%globalConfig.dataset)

CreateParam = lasagne.utils.create_param
InitW = lasagne.init.Normal(std=0.02, mean=0.)
InitGamma = lasagne.init.Normal(std=0.02, mean=1.)
InitBeta = lasagne.init.Constant(0.)

class ForwardRender(object):
    nyu_centerPtIdx = 30
    nyu_palmPtIdx = [31,32,33,34,35]

    msra_centerPtIdx = 0
    msra_palmPtIdx = [1,5,9,13,17]

    icl_centerPtIdx = 0
    icl_palmPtIdx = [1,4,7,10,13]

    def __init__(self, x_dim):
        # initialize the translation and orientation input layer
        self.origin_input_var = T.fmatrix('origin_input')
        self.origin_input_layer = lasagne.layers.InputLayer(shape=(None, 3),
                                                            input_var =\
                                                            self.origin_input_var)

        # build the network
        self.pose_vae = PoseVAE(x_dim=x_dim)
        self.alignment_layer = \
                self.build_latent_alignment_layer(self.pose_vae,
                                                  self.origin_input_layer)
        self.alignment_var = lasagne.layers.get_output(self.alignment_layer,
                                                       self.latent_var,
                                                       deterministic=False)
        self.alignment_tvar = lasagne.layers.get_output(self.alignment_layer,
                                                       self.latent_tvar,
                                                       deterministic=True)

        self.depth_gan = DepthGAN(z_dim=self.z_dim)
        self.render_layer = self.depth_gan.gen_depth_layer
        self.render_var = lasagne.layers.get_output(self.render_layer,
                                                   self.alignment_var,
                                                   deterministic=False)
        self.render_tvar = lasagne.layers.get_output(self.render_layer,
                                                   self.alignment_tvar,
                                                   deterministic=True)

        # hyper training parameters
        self.lr, self.b1 = 0.001, 0.5
        self.batch_size = 200
        print 'vae and gan initialized'

        self.params = self.pose_vae.encoder_params +\
                self.alignment_params +\
                self.depth_gan.gen_params
        print 'all parameters: {}'.format(self.params)
    
    def build_latent_alignment_layer(self, pose_vae, \
                                     origin_layer = None,\
                                     quad_layer = None):
        self.pose_z_dim = lasagne.layers.get_output_shape(pose_vae.z_layer)[1]
        self.z_dim = self.pose_z_dim
        if origin_layer is not None:
            self.z_dim += 3
        if quad_layer is not None:
            self.z_dim += 4

        align_w = CreateParam(InitW, 
                              (self.z_dim, self.z_dim), 
                              'align_w')
        align_b = CreateParam(InitBeta, 
                              (self.z_dim,), 
                              'align_b')
        align_g = CreateParam(InitGamma, 
                              (self.z_dim,), 
                              'align_g')

        latent_layer = pose_vae.z_layer
        if origin_layer is not None:
            latent_layer = lasagne.layers.ConcatLayer([latent_layer,
                                                      self.origin_input_layer],
                                                     axis = 1)
        if quad_layer is not None:
            latent_layer = lasagne.layers.ConcatLayer([latent_layer,
                                                      quad_layer],
                                                      axis = 1)

        print 'latent_layer output shape = {}'\
                .format(lasagne.layers.get_output_shape(latent_layer))
        self.latent_layer = latent_layer
        self.latent_var = lasagne.layers.get_output(self.latent_layer,
                                                    deterministic=False)
        self.latent_tvar = lasagne.layers.get_output(self.latent_layer,
                                                    deterministic=True)

        # use None input, to adapt z from both pose-vae and real-test
        latent_layer = lasagne.layers.InputLayer(shape=(None,self.z_dim))

        alignment_layer = batch_norm(
            lasagne.layers.DenseLayer(latent_layer,
                                      num_units = self.z_dim,
                                      nonlinearity=None,
                                      W=align_w),
            beta=align_b, gamma=align_g)

        self.alignment_params = [align_w, align_b, align_g]
        nPara = len(self.alignment_params) + 2
        self.alignment_all_params =\
                lasagne.layers.get_all_params(alignment_layer)[-nPara:]
        return alignment_layer

    def prepareData(self, ds):
        self.ds = ds
        self.ds.normTranslation()
        self.ds.normRotation()
        self.ds.frmToNp()

        self.data_depth = self.ds.x_norm
        self.data_depth *= np.float32(2)
        print 'depth range={} to {}'.format(self.data_depth.min(),
                                            self.data_depth.max())
        self.data_pose_skel = self.ds.y_norm # skeleton in conanical view

        self.ndata = len(self.data_pose_skel)

        self.data_pose_orig = np.zeros((self.ndata, 3), np.float32)
        self.data_pose_quad = np.zeros((self.ndata, 4), np.float32)
        for i,frm in enumerate(self.ds.frmList):
            self.data_pose_orig[i] = frm.origin
            self.data_pose_quad[i] = frm.quad
        print 'data prepared'

    def resumePose(self, norm_pose, tran, quad=None):
        orig_pose = norm_pose.copy()
        orig_pose.shape = (-1,3)
        if quad is not None:
            R = Matrix33(quad)
            orig_pose = np.dot(R.transpose(), orig_pose.transpose())
        translation = repmat(tran.reshape((1,3)), orig_pose.shape[0], 1)
        orig_pose = translation + orig_pose
        orig_pose = orig_pose.flatten()
        return orig_pose

    def visPair(self, depth, pose=None, trans=None, com=None, ratio=None):
        img = depth[0].copy()
        img = (img+1)*127.0
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
        if pose is None:
            return img
        
        skel = pose.copy()
        skel.shape = (-1, 3)
        skel = skel*ratio
        skel2 = []
        for pt in skel:
            pt2 = Camera.to2D(pt+com)
            pt2[2] = 1.0
            pt2 = np.dot(trans, pt2)
            pt2.shape = (3,1)
            pt2 = (pt2[0],pt2[1])
            skel2.append(pt2)
        for idx, pt2 in enumerate(skel2):
            cv2.circle(img, pt2, 3, 
                       data.util.figColor[colorPlatte[idx]], -1)
        for b in bones:
            pt1 = skel2[b[0]]
            pt2 = skel2[b[1]]
            color = b[2]
            cv2.line(img,pt1,pt2,color,2)
        return img

    def genLossAndGradient(self):
        #establish loss
        self.pose_input_var = self.pose_vae.pose_input_var
        self.noise_input_var = self.pose_vae.noise_input_var
        self.real_depth_var = T.ftensor4('real_depth')
        self.pixel_loss = lasagne.objectives.squared_error(self.render_var,
                                                self.real_depth_var)
        self.pixel_loss = lasagne.objectives.aggregate(self.pixel_loss, 
                                                      mode='mean')

        #calculate gradient
        print 'param: {}'.format(self.params)
        self.updates = lasagne.updates.adam(self.pixel_loss, self.params,
                                            self.lr, self.b1)
        #compile function
        self.train_fn = theano.function(
            [self.pose_input_var, 
             self.origin_input_var,
             # self.quad_input_var,
             self.noise_input_var, 
             self.real_depth_var],
            self.pixel_loss,
            updates = self.updates
        )
        self.render_fn = theano.function(
            [self.pose_input_var,
             self.origin_input_var,
             # self.quad_input_var,
             self.noise_input_var],
            self.render_tvar
        )

        updates = lasagne.updates.adam(self.pixel_loss, self.alignment_params,
                                      self.lr, self.b1) 
        self.alignment_train_fn = theano.function(
            [self.pose_input_var, 
             self.origin_input_var,
             # self.quad_input_var,
             self.noise_input_var, 
             self.real_depth_var],
            self.pixel_loss,
            updates = updates
        )
        print 'function compiled'

    def saveAlignParam(self, path):
        align_param_vals = [param.get_value() for param in self.alignment_all_params]
        align_path = path+'_align.npz'
        np.savez(align_path, *align_param_vals)

    def loadAlignParam(self, path):
        align_path = path+'_align.npz'
        align = np.load(align_path)
        for idx, param in enumerate(self.alignment_all_params):
            param.set_value(align['arr_%d'%idx])

    def saveParam(self, path):
        pass

    def loadParam(self, path):
        pass

    def train(self, nepoch, desc='dummy'):
        cache_dir = os.path.join(globalConfig.model_dir, 'render/%s_%s'%(globalConfig.dataset,desc))

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        img_dir = os.path.join(cache_dir, 'img')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        tr_stream = MultiDataStream([self.data_pose_skel, 
                                     self.data_pose_orig,
                                     self.data_pose_quad,
                                     self.data_depth])

        val_data = [self.data_pose_skel[0:self.batch_size], 
                    self.data_pose_orig[0:self.batch_size],
                    self.data_pose_quad[0:self.batch_size],
                    np.zeros((self.batch_size, self.pose_z_dim), np.float32)]


        print '[ForwardRender] enter training loop with %d epoches'%nepoch
        seed = 42
        np_rng = RandomState(seed)

        for epoch in range(nepoch):
            tr_err, nupdates = 0, 0
            start_time = time.time()
            for data in tr_stream.iterate(batchsize = self.batch_size):
                noise = np.zeros((self.batch_size, self.pose_z_dim), np.float32)
                skel, orig, quad, depth = data
                tr_err += self.train_fn(skel, orig, quad, noise, depth)
                nupdates += 1

            print 'Epoch {} of {} took {:.3f}s'.format(epoch, nepoch,
                                                      time.time()- start_time)
            print 'loss = {}'.format(tr_err/nupdates)

            if epoch % 10 == 0:
               recons_depth = self.render_fn(*val_data) 
               for idx in range(0, len(recons_depth), 10):
                   img = self.depth_gan.vis_depth(recons_depth[idx])
                   cv2.imwrite(os.path.join(img_dir, '%d_%d.jpg'%(epoch, idx)), 
                               img)


