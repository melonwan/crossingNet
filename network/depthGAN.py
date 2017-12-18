import globalConfig
from data.dataset import *
from data.util import Camera
from data.stream import DataStream

import lasagne, theano, cv2, time
from lasagne.layers import batch_norm
from numpy.random import RandomState
import theano.tensor as T
import shutil
from sklearn.externals import joblib
import json

lr = 0.0002
b1 = 0.5
noise_dim = 20
K = 1 

CreateParam = lasagne.utils.create_param
InitW = lasagne.init.Normal(std=0.05, mean=0.)
InitGamma = lasagne.init.Normal(std=0.02, mean=1.)
InitBeta = lasagne.init.Constant(0.)
lrelu = lasagne.nonlinearities.LeakyRectify(leakiness=0.2)

class DepthGAN(object):
    def __init__(self, z_dim, batch_size = 100, lr = 0.0005, b1 =0.5):
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.z_std = 0.6 # used when z is normal distribution
        print 'depthGAN is initialized with z_dim=%d'%self.z_dim

        # build network
        self.noise_input_layer = lasagne.layers.InputLayer((None, self.z_dim))
        self.gen_depth_layer = \
            self.build_generative(self.noise_input_layer)
        self.depth_shape =\
            lasagne.layers.get_output_shape(self.gen_depth_layer)
        print 'gen build with generated depth shape={}'.format(self.depth_shape)
        self.build_discriminative()
    
    def build_generative(self, noise_input_layer):
        #firstly create the network shapes and weights
        gg_input = CreateParam(InitGamma, (self.z_dim,), 'gg_input')
        gb_input = CreateParam(InitBeta, (self.z_dim,), 'gb_input')

        full_shape = (self.z_dim, 32*4*4)
        gw_full = CreateParam(InitW, full_shape, 'gw_full')
        gg_full = CreateParam(InitGamma, (full_shape[1],), 'gg_full')
        gb_full = CreateParam(InitBeta, (full_shape[1],), 'gb_full')

        deconv1_shape = (32, 32, 6, 6)
        gw1 = CreateParam(InitW, deconv1_shape, 'gw1')
        gg1 = CreateParam(InitGamma, (deconv1_shape[1],), 'gg1')
        gb1 = CreateParam(InitBeta, (deconv1_shape[1],), 'gb1')

        deconv2_shape = (32, 32, 6, 6)
        gw2 = CreateParam(InitW, deconv2_shape, 'gw2')
        gg2 = CreateParam(InitGamma, (deconv2_shape[1],), 'gg2')
        gb2 = CreateParam(InitBeta, (deconv2_shape[1],), 'gb2')

        deconv3_shape = (32, 32, 6, 6)
        gw3 = CreateParam(InitW, deconv3_shape, 'gw3')
        gg3 = CreateParam(InitGamma, (deconv3_shape[1],), 'gg3')
        gb3 = CreateParam(InitBeta, (deconv3_shape[1],), 'gb3')

        deconv4_shape = (32, 32, 6, 6)
        gw4 = CreateParam(InitW, deconv4_shape, 'gw4')
        gg4 = CreateParam(InitGamma, (deconv4_shape[1],), 'gg4')
        gb4 = CreateParam(InitBeta, (deconv4_shape[1],), 'gb4')

        deconv5_shape = (32, 1, 6, 6)
        gw5 = CreateParam(InitW, deconv5_shape, 'gw5')
        gb5 = CreateParam(lasagne.init.Constant(0.8), (deconv5_shape[1],), 'gb5')

        # dense layer
        noise_input_layer = lasagne.layers.NonlinearityLayer(noise_input_layer,
                                                            nonlinearity=lasagne.nonlinearities.tanh)
        dense_1 = batch_norm(lasagne.layers.DenseLayer(noise_input_layer,
                                                       num_units=32*4*4,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            W=gw_full),
                            beta=gb_full, gamma=gg_full)
        deconv_0 = lasagne.layers.ReshapeLayer(dense_1, ([0], 32, 4, 4))

        #deconv layer
        deconv_1 = batch_norm(lasagne.layers.TransposedConv2DLayer(deconv_0,
                                                                   num_filters=deconv1_shape[1],
                                                                   W=gw1,
                                                                   filter_size = 6,
                                                                   stride = 2,
                                                                   crop = 2,
                                                                   nonlinearity=lasagne.nonlinearities.rectify),
                             beta=gb1, gamma=gg1)

        deconv_2 = batch_norm(lasagne.layers.TransposedConv2DLayer(deconv_1, 
                                                                   num_filters=deconv2_shape[1],
                                                                   W=gw2,
                                                                   filter_size = 6,
                                                                   stride = 2, 
                                                                   crop = 2,
                                                                   nonlinearity=lasagne.nonlinearities.rectify),
                             beta=gb2, gamma=gg2)

        deconv_3 = batch_norm(lasagne.layers.TransposedConv2DLayer(deconv_2, 
                                                                   num_filters=deconv3_shape[1],
                                                                   W=gw3,
                                                                   b=None,
                                                                   filter_size = 6,
                                                                   stride = 2,
                                                                   crop = 2,
                                                                   nonlinearity=lasagne.nonlinearities.rectify),
                              beta=gb3, gamma=gg3)

        deconv_4 = batch_norm(lasagne.layers.TransposedConv2DLayer(deconv_3, 
                                                                   num_filters=deconv4_shape[1],
                                                                   W=gw4,
                                                                   b=None,
                                                                   filter_size = 6,
                                                                   stride = 2,
                                                                   crop = 2,
                                                                   nonlinearity=lasagne.nonlinearities.rectify),
                             beta=gb4, gamma=gg4)

        deconv_5 = lasagne.layers.TransposedConv2DLayer(deconv_4, 
                                                       num_filters=deconv5_shape[1],
                                                       W=gw5,
                                                       b=gb5, 
                                                       filter_size = 6,
                                                       stride = 2,
                                                       crop = 2,
                                                       nonlinearity=lasagne.nonlinearities.tanh)
        self.gen_params = [ #gb_input, gg_input,
                           gw_full, gb_full, gg_full, 
                           gw1,gb1, gg1,
                           gw2, gb2, gg2, 
                           gw3, gb3, gg3,
                           gw4, gb4, gg4,
                           gw5, gb5]
        nPara = len(self.gen_params) + 2*5
        self.gen_all_params = lasagne.layers.get_all_params(deconv_5)[-nPara:]
        return deconv_5

    def build_discriminative(self):
        # firstly create the network shape and weights
        conv1_shape = (32, 1, 6, 6)
        dw1 = CreateParam(InitW, conv1_shape, 'dw1')
        dg1 = CreateParam(InitGamma, (conv1_shape[0],), 'dg1')
        db1 = CreateParam(InitBeta, (conv1_shape[0],), 'db1')

        conv2_shape = (32, 32, 6 ,6)
        dw2 = CreateParam(InitW, conv2_shape, 'dw2')
        dg2 = CreateParam(InitGamma, (conv2_shape[0],), 'dg2')
        db2 = CreateParam(InitBeta, (conv2_shape[0],), 'db2')

        conv3_shape = (32, 32, 6, 6)
        dw3 = CreateParam(InitW, conv3_shape, 'dw3')
        dg3 = CreateParam(InitGamma, (conv3_shape[0],), 'dg3')
        db3 = CreateParam(InitBeta, (conv3_shape[0],), 'db3')

        conv4_shape = (32, 32, 6, 6)
        dw4 = CreateParam(InitW, conv4_shape, 'dw4')
        dg4 = CreateParam(InitGamma, (conv4_shape[0],), 'dg4')
        db4 = CreateParam(InitBeta, (conv4_shape[0],), 'db4')

        conv5_shape = (32, 32, 6, 6)
        dw5 = CreateParam(InitW, conv5_shape, 'dw5')
        dg5 = CreateParam(InitGamma, (conv5_shape[0],), 'dg5')
        db5 = CreateParam(InitBeta, (conv5_shape[0],), 'db5')

        full_shape = (32*4*4, 1)
        dw_full = CreateParam(InitW, full_shape, 'dw_full')
        db_full = CreateParam(InitBeta, (full_shape[1],), 'db_full')

        # build dis net
        dis_input = lasagne.layers.InputLayer(shape=(None, 1, 128, 128))
        self.dis_render_layer = dis_input
        dis_conv1 = batch_norm(lasagne.layers.Conv2DLayer(dis_input,
                                                num_filters=conv1_shape[0],
                                                filter_size=(conv1_shape[2],conv1_shape[3]),
                                                stride=2,
                                                pad=2,
                                                W=dw1,
                                                b=None,
                                                nonlinearity=lrelu),
                                gamma=dg1, beta=db1)

        dis_conv2 = batch_norm(lasagne.layers.Conv2DLayer(dis_conv1,
                                                num_filters=conv2_shape[0],
                                                filter_size=(conv2_shape[2],conv2_shape[3]),
                                                stride=2,
                                                pad=2,
                                                W=dw2,
                                                b=None,
                                                nonlinearity=lrelu),
                                gamma=dg2, beta=db2)
        self.dis_conv2 = dis_conv2

        # to be later used for recognition or other discriminative purpose
        self.dis_hidden_layer = dis_conv2
        self.dis_metric_layer = dis_conv2

        dis_conv3 = batch_norm(lasagne.layers.Conv2DLayer(dis_conv2,
                                                num_filters=conv3_shape[0],
                                                filter_size=(conv3_shape[2],conv3_shape[3]),
                                                stride=2,
                                                pad=2,
                                                W=dw3,
                                                b=None,
                                                nonlinearity=lrelu),
                                gamma=dg3, beta=db3)
        self.dis_conv3 = dis_conv3

        dis_conv4 = batch_norm(lasagne.layers.Conv2DLayer(dis_conv3,
                                                num_filters=conv4_shape[0],
                                                filter_size=(conv4_shape[2],conv4_shape[3]),
                                                stride=2,
                                                pad=2,
                                                W=dw4,
                                                b=None,
                                                nonlinearity=lrelu),
                               gamma=dg4, beta=db4)
        self.dis_conv4 = dis_conv4

        dis_conv5 = batch_norm(lasagne.layers.Conv2DLayer(dis_conv4,
                                                num_filters=conv5_shape[0],
                                                filter_size=(conv5_shape[2],conv5_shape[3]),
                                                stride=2,
                                                pad=2,
                                                W=dw5,
                                                b=None,
                                                nonlinearity=lrelu),
                               gamma=dg5, beta=db5)
        self.feamat_layer = dis_conv5
        self.dis_conv5 = dis_conv5

        dis_full = lasagne.layers.DenseLayer(dis_conv5,
                                              num_units=full_shape[1], 
                                              W=dw_full,
                                              b=db_full,
                                              nonlinearity=lasagne.nonlinearities.sigmoid)
        self.dis_px_layer = dis_full
        self.dis_params = [dw1, db1, dg1,
                           dw2, db2, dg2,
                           dw3, db3, dg3,
                           dw4, db4, dg4,
                           dw5, db5, dg5,
                           dw_full, db_full]
        nPara = len(self.dis_params) + 2*5 
        self.dis_all_params = lasagne.layers.get_all_params(dis_full)[-nPara:]

    def build_metric(self, output_dim, hidden_layer=None):
        if hidden_layer is None:
            hidden_layer = self.dis_metric_layer

        conv1_shape = (32, 32, 6, 6)
        dw1 = CreateParam(InitW, conv1_shape, 'mdw1')
        dg1 = CreateParam(InitGamma, (conv1_shape[0],), 'mdg1')
        db1 = CreateParam(InitBeta, (conv1_shape[0],), 'mdb1')

        conv2_shape = (32, 32, 6 ,6)
        dw2 = CreateParam(InitW, conv2_shape, 'mdw2')
        dg2 = CreateParam(InitGamma, (conv2_shape[0],), 'mdg2')
        db2 = CreateParam(InitBeta, (conv2_shape[0],), 'mdb2')

        conv3_shape = (32, 32, 6, 6)
        dw3 = CreateParam(InitW, conv3_shape, 'mdw3')
        dg3 = CreateParam(InitGamma, (conv3_shape[0],), 'mdg3')
        db3 = CreateParam(InitBeta, (conv3_shape[0],), 'mdb3')

        conv1 = batch_norm(lasagne.layers.Conv2DLayer(
            hidden_layer,
            num_filters=conv1_shape[0],
            filter_size=(conv1_shape[2],conv1_shape[3]),
            pad=2,
            stride=2,
            W=dw1,
            b=None,
            nonlinearity=lrelu),
            gamma=dg1, beta=db1)
        conv2 = batch_norm(lasagne.layers.Conv2DLayer(
            conv1,
            num_filters=conv2_shape[0],
            filter_size=(conv2_shape[2],conv2_shape[3]),
            pad=2,
            stride=2,
            W=dw2,
            b=None,
            nonlinearity=lrelu),
            gamma=dg2, beta=db2)
        conv3 = batch_norm(lasagne.layers.Conv2DLayer(
            conv2,
            num_filters=conv3_shape[0],
            filter_size=(conv3_shape[2],conv3_shape[3]),
            pad=2,
            stride=2,
            W=dw3,
            b=None,
            nonlinearity=lrelu),
            gamma=dg3, beta=db3)

        # fully connected part
        input_dim = lasagne.layers.get_output_shape(conv3)
        print '[DepthGAN][build_metric hidden] conv_output_shape=', input_dim
        input_dim = input_dim[1]*input_dim[2]*input_dim[3]
        hidden_dim = 512 
        metric_shape_1 = (input_dim, hidden_dim)
        dw_metric_1 = CreateParam(InitW, metric_shape_1, 'dw_metric_1')
        db_metric_1 = CreateParam(InitBeta, (metric_shape_1[1],), 'db_metric_1')
        dg_metric_1 = CreateParam(InitGamma, (metric_shape_1[1],), 'dg_metric_1')
        
        metric_shape_2 = (hidden_dim, output_dim)
        dw_metric_2 = CreateParam(InitW, metric_shape_2, 'dw_metric_2')
        db_metric_2 = CreateParam(InitBeta, (metric_shape_2[1],), 'db_metric_2')

        metric_layer = batch_norm(lasagne.layers.DenseLayer(conv3,
                                num_units = metric_shape_1[1],
                                W = dw_metric_1,
                                nonlinearity=lrelu),
                                beta=db_metric_1, gamma=dg_metric_1)
        metric_layer = lasagne.layers.DenseLayer(metric_layer,
                                              num_units = metric_shape_2,
                                              W = dw_metric_2,
                                              b = db_metric_2,
                                              nonlinearity=None)

        self.metric_params = [
                           dw1, db1, dg1,
                           dw2, db2, dg2, 
                           dw3, db3, dg3,
                           dw_metric_1, db_metric_1, dg_metric_1, 
                           dw_metric_2, db_metric_2]
        nPara = len(self.metric_params) + 2*4
        self.metric_all_params = lasagne.layers.get_all_params(metric_layer)[-nPara:]
        return metric_layer

    def build_metric_combi(self, output_dim, hidden_layer=None):
        if hidden_layer is None:
            hidden_layer = self.dis_metric_layer

        conv1_shape = (32, 32, 6, 6)
        dw1 = CreateParam(InitW, conv1_shape, 'mdw1')
        dg1 = CreateParam(InitGamma, (conv1_shape[0],), 'mdg1')
        db1 = CreateParam(InitBeta, (conv1_shape[0],), 'mdb1')

        conv2_shape = (32, 32, 6 ,6)
        dw2 = CreateParam(InitW, conv2_shape, 'mdw2')
        dg2 = CreateParam(InitGamma, (conv2_shape[0],), 'mdg2')
        db2 = CreateParam(InitBeta, (conv2_shape[0],), 'mdb2')

        conv3_shape = (32, 32, 6, 6)
        dw3 = CreateParam(InitW, conv3_shape, 'mdw3')
        dg3 = CreateParam(InitGamma, (conv3_shape[0],), 'mdg3')
        db3 = CreateParam(InitBeta, (conv3_shape[0],), 'mdb3')

        conv1 = batch_norm(lasagne.layers.Conv2DLayer(
            hidden_layer,
            num_filters=conv1_shape[0],
            filter_size=(conv1_shape[2],conv1_shape[3]),
            pad=2,
            stride=2,
            W=dw1,
            b=None,
            nonlinearity=lrelu),
            gamma=dg1, beta=db1)
        conv2 = batch_norm(lasagne.layers.Conv2DLayer(
            conv1,
            num_filters=conv2_shape[0],
            filter_size=(conv2_shape[2],conv2_shape[3]),
            pad=2,
            stride=2,
            W=dw2,
            b=None,
            nonlinearity=lrelu),
            gamma=dg2, beta=db2)
        conv3 = batch_norm(lasagne.layers.Conv2DLayer(
            conv2,
            num_filters=conv3_shape[0],
            filter_size=(conv3_shape[2],conv3_shape[3]),
            pad=2,
            stride=2,
            W=dw3,
            b=None,
            nonlinearity=lrelu),
            gamma=dg3, beta=db3)

        # fully connected part
        input_dim = lasagne.layers.get_output_shape(conv3)
        print '[DepthGAN][build_metric hidden] conv_output_shape=', input_dim
        input_dim = input_dim[1]*input_dim[2]*input_dim[3]
        hidden_dim = 512 
        metric_shape_1 = (input_dim, hidden_dim)
        dw_metric_1 = CreateParam(InitW, metric_shape_1, 'dw_metric_1')
        db_metric_1 = CreateParam(InitBeta, (metric_shape_1[1],), 'db_metric_1')
        metric_layer = lasagne.layers.DenseLayer(conv3,
                                num_units = metric_shape_1[1],
                                W = dw_metric_1,
                                b = db_metric_1,
                                nonlinearity=lrelu)
        
        metric_shape_2 = (2*hidden_dim, output_dim)
        dw_metric_2 = CreateParam(InitW, metric_shape_2, 'dw_metric_2')
        db_metric_2 = CreateParam(InitBeta, (metric_shape_2[1],), 'db_metric_2')
        
        input_layer = lasagne.layers.InputLayer(shape=(None,2*hidden_dim))
        combi_metric_layer = lasagne.layers.DenseLayer(input_layer,
                                              num_units = metric_shape_2,
                                              W = dw_metric_2,
                                              b = db_metric_2,
                                              nonlinearity=None)

        self.metric_params = [
                             dw1, db1, dg1,
                             dw2, db2, dg2, 
                             dw3, db3, dg3,
                             dw_metric_1, db_metric_1, 
                             dw_metric_2, db_metric_2]
        nPara = len(self.metric_params) + 2*3
        self.metric_all_params = lasagne.layers.get_all_params(metric_layer)[-nPara:]
        return metric_layer, combi_metric_layer

    def build_recognition(self, output_dim, hidden_layer=None):
        # this sub-network is used for ICVL and MSRA dastaset. 
        # more complicated network architecture is used for NYU.
        if globalConfig.dataset == 'NYU':
            raise NotImplementedError

        if hidden_layer is None:
            hidden_layer = self.dis_hidden_layer

        conv1_shape = (32, 32, 6, 6)
        dw1 = CreateParam(InitW, conv1_shape, 'rdw1')
        dg1 = CreateParam(InitGamma, (conv1_shape[0],), 'rdg1')
        db1 = CreateParam(InitBeta, (conv1_shape[0],), 'rdb1')

        conv2_shape = (32, 32, 6 ,6)
        dw2 = CreateParam(InitW, conv2_shape, 'rdw2')
        dg2 = CreateParam(InitGamma, (conv2_shape[0],), 'rdg2')
        db2 = CreateParam(InitBeta, (conv2_shape[0],), 'rdb2')

        conv3_shape = (32, 32, 6, 6)
        dw3 = CreateParam(InitW, conv3_shape, 'rdw3')
        dg3 = CreateParam(InitGamma, (conv3_shape[0],), 'rdg3')
        db3 = CreateParam(InitBeta, (conv3_shape[0],), 'rdb3')

        conv1 = batch_norm(lasagne.layers.Conv2DLayer(
            hidden_layer,
            num_filters=conv1_shape[0],
            filter_size=(conv1_shape[2],conv1_shape[3]),
            stride=2,
            pad=2,
            W=dw1,
            b=None,
            nonlinearity=lrelu),
            gamma=dg1, beta=db1)
        conv2 = batch_norm(lasagne.layers.Conv2DLayer(
            conv1,
            num_filters=conv2_shape[0],
            filter_size=(conv2_shape[2],conv2_shape[3]),
            stride=2,
            pad=2,
            W=dw2,
            b=None,
            nonlinearity=lrelu),
            gamma=dg2, beta=db2)
        conv3 = batch_norm(lasagne.layers.Conv2DLayer(
            conv2,
            num_filters=conv3_shape[0],
            filter_size=(conv3_shape[2],conv3_shape[3]),
            stride=2,
            pad=2,
            W=dw3,
            b=None,
            nonlinearity=lrelu),
            gamma=dg3, beta=db3)

        # fully connected part
        input_dim = lasagne.layers.get_output_shape(conv3)
        print '[DepthGAN][build_recognition hidden] conv_output_shape=', input_dim
        input_dim = input_dim[1]*input_dim[2]*input_dim[3]
        reco_shape = (input_dim, output_dim)
        dw_reco_1 = CreateParam(InitW, reco_shape, 'dw_reco_1')
        db_reco_1 = CreateParam(InitBeta, (reco_shape[1],), 'db_reco_1')
        dg_reco_1 = CreateParam(InitGamma, (reco_shape[1],), 'dg_reco_1')
        
        dw_reco_2 = CreateParam(InitW, (output_dim, output_dim), 'dw_reco_2')
        db_reco_2 = CreateParam(InitBeta, (reco_shape[1],), 'db_reco_2')

        reco_layer = batch_norm(lasagne.layers.DenseLayer(conv3,
                                num_units = reco_shape[1],
                                W = dw_reco_1,
                                nonlinearity=lrelu),
                                beta=db_reco_1, gamma=dg_reco_1)
        reco_layer = lasagne.layers.DenseLayer(reco_layer,
                                              num_units = output_dim,
                                              W = dw_reco_2,
                                              b = db_reco_2,
                                              nonlinearity=None)

        self.reco_params = [
                           dw1, db1, dg1,
                           dw2, db2, dg2, 
                           dw3, db3, dg3,
                           dw_reco_1, db_reco_1, dg_reco_1, 
                           dw_reco_2, db_reco_2]
        nPara = len(self.reco_params) + 2*4
        self.reco_all_params = lasagne.layers.get_all_params(reco_layer)[-nPara:]
        return reco_layer

    def linkSubNets(self, noiseInputVar=None):
        # for every subnet, the input is None
        if noiseInputVar is None:
            noiseInputVar = T.fmatrix('noise_input')

        self.noise_input_var = noiseInputVar
        self.depth_input_var = T.ftensor4('real_depth')
        self.gen_depth_var = lasagne.layers.get_output(self.gen_depth_layer,
                                                       self.noise_input_var,
                                                       deterministic=False)
        self.gen_depth_tvar = lasagne.layers.get_output(self.gen_depth_layer,
                                                        self.noise_input_var,
                                                       deterministic=True)
        real_var = self.depth_input_var
        fake_var = self.gen_depth_var

        self.real_feamat_var=T.mean(lasagne.layers.get_output(self.feamat_layer,
                                                                real_var),
                                      axis=0)
        self.fake_feamat_var=T.mean(lasagne.layers.get_output(self.feamat_layer,
                                                                fake_var),
                                      axis=0)
        self.px_real_var = lasagne.layers.get_output(self.dis_px_layer,
                                                     real_var) 
        self.px_fake_var = lasagne.layers.get_output(self.dis_px_layer,
                                                     fake_var)

    def genLossAndGradient(self):
        self.linkSubNets()
        # establish objectives
        loss_gen = T.mean(abs(self.real_feamat_var- self.fake_feamat_var)) 
        self.loss_gen = lasagne.objectives.aggregate(loss_gen, mode='mean')

        loss_dis_fake = lasagne.objectives.binary_crossentropy(self.px_fake_var,
                                                               T.zeros(self.px_fake_var.shape))
        loss_dis_fake = lasagne.objectives.aggregate(loss_dis_fake, mode='mean')

        loss_dis_real = lasagne.objectives.binary_crossentropy(self.px_real_var,
                                                               T.ones(self.px_real_var.shape))
        loss_dis_real = lasagne.objectives.aggregate(loss_dis_real, mode='mean')
        
        self.loss_dis = loss_dis_fake + loss_dis_real
        
        # calculate gradients
        self.gen_update_var = lasagne.updates.adam(self.loss_gen,
                                                   self.gen_params, 
                                                   self.lr,
                                                   self.b1)
        self.dis_update_var = lasagne.updates.adam(self.loss_dis,
                                                   self.dis_params, 
                                                   self.lr,
                                                   self.b1)

        # compile function
        print '[depthGAN] begin compling ...'
        self.train_gen_fn = theano.function([self.noise_input_var,
                                             self.depth_input_var], 
                                            self.loss_gen,
                                            updates=self.gen_update_var)
        print 'train_gen_fn compiled'
        self.train_dis_fn = theano.function([self.noise_input_var, 
                                             self.depth_input_var],
                                            self.loss_dis,
                                            updates=self.dis_update_var)
        print 'train_dis_fn compiled'
        self.gen_fn = theano.function([self.noise_input_var],
                                         self.gen_depth_tvar)
        print 'gen_fn compiled'
        print 'all functions compiled'

    def saveParam(self, path):
        gen_param_vals = [param.get_value() for param in self.gen_all_params]
        dis_param_vals = [param.get_value() for param in self.dis_all_params]

        gen_path = path+'_gen_params.npz'
        np.savez(gen_path, *gen_param_vals)
        
        dis_path = path+'_dis_params.npz'
        np.savez(dis_path, *dis_param_vals)

        if hasattr(self, 'reco_all_params'):
            rec_param_vals = [param.get_value() for param in
                              self.reco_all_params]
            reco_path = path+'_reco_params.npz'
            np.savez(reco_path, *rec_param_vals)

        if hasattr(self, 'metric_all_params'):
            metr_param_vals = [param.get_value() for param in
                              self.metric_all_params]
            metr_path = path+'_metr_params.npz'
            np.savez(metr_path, *metr_param_vals)
        print '[depthGAN] parameters has been saved to %s_gen/dis_params.npz'%path

    def loadParam(self, path):
        gen_path = path+'_gen_params.npz'
        gen = np.load(gen_path)
        for idx, param in enumerate(self.gen_all_params):
            param.set_value(gen['arr_%d'%idx])
        
        dis_path = path+'_dis_params.npz'
        dis = np.load(dis_path)
        for idx, param in enumerate(self.dis_all_params):
            param.set_value(dis['arr_%d'%idx])

        if hasattr(self, 'reco_all_params'):
            reco_path = path+'_reco_params.npz'
            if not os.path.exists(reco_path):
                print '[WARN] %s is not existed, not influence training'%reco_path
            else:
                reco = np.load(reco_path)
                for idx, param in enumerate(self.reco_all_params):
                    param.set_value(reco['arr_%d'%idx])

        if hasattr(self, 'metric_all_params'):
            metr_path = path+'_metr_params.npz'
            if not os.path.exists(metr_path):
                print '[WARN] %s is not existed, not influence training'%metr_path
            else:
                metr = np.load(metr_path)
                for idx, param in enumerate(self.metric_all_params):
                    param.set_value(metr['arr_%d'%idx])
        print '[depghGAN] parameters have been loaded from %s_gen/dis_params.npz'%path

    def train(self, train_stream, desc, nepoch=0, isVal=True,\
              isFeaMat=False, isRm=False):
        cache_dir = os.path.join(globalConfig.model_dir, 'depth_gan/%s_%s'%(globalConfig.dataset,desc))
        if isRm and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        img_dir = os.path.join(cache_dir, 'img/')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        params_dir = os.path.join(cache_dir, 'params/')
        if not os.path.exists(params_dir):
            os.mkdir(params_dir)

        log_path = os.path.join(cache_dir, 'log.txt')
        logf = open(log_path, 'w')
        logf.close()

        seed = 42
        np_rng = RandomState(seed)
        print '[DepthGAN] enter the training loop with %d epoches'%nepoch

        val_noises = np_rng.uniform(-1, 1, (self.batch_size, self.z_dim)).astype(np.float32)
        
        for epoch in range(nepoch):
            tr_err, gen_err, dis_err = (0,)*3
            nupdates, dis_nupdates, gen_nupdates = (0,)*3
            start_time = time.time()

            for depth in train_stream.iterate(batchsize=self.batch_size,
                                              shuffle=True):
                noises = np_rng.uniform(-1.,1., (self.batch_size, self.z_dim)).astype(np.float32)

                dis_err += self.train_dis_fn(noises, depth)
                dis_nupdates += 1

                if not isFeaMat:
                    gen_err += self.train_gen_fn(noises)
                else:
                    gen_err += self.train_gen_fn(noises, depth)
                gen_nupdates += 1
                nupdates += 1

            if epoch%10 == 0:
                gen_res = self.gen_fn(val_noises)
                for noise_idx in range(min(10, len(gen_res))):
                    img = self.vis_depth(gen_res[noise_idx])
                    cv2.imwrite('%s/%d_%d.jpg'%(img_dir,
                                                epoch,
                                                noise_idx), 
                                img)

            if epoch %10 == 0:
                self.saveParam('%s%d'%(params_dir, -1))
            if epoch %100 == 0:
                self.saveParam('%s%d'%(params_dir, epoch))

            print "Epoch {} of {} took {:.3f}s".format(epoch+1, nepoch,
                                                       time.time()-start_time)
            print "DisLoss = {:.6f}, GenLoss = {:.6f}".format(dis_err/dis_nupdates,
                                                             gen_err/gen_nupdates)
            flog = open(log_path, 'a')
            flog.write(json.dumps((dis_err/dis_nupdates, gen_err/gen_nupdates))+'\n')
            flog.write('epoch=%d, time=%f s'%(epoch, time.time()-start_time))
            flog.close()
            
    def vis_depth(self,normed_vec):
        img = normed_vec.copy()
        img.shape = (128, 128)
        img = (img+1.0)*127.0
        img = img.astype('uint8') 
        return img

log_fields = [
    'n_epochs',
    'g_cost',
    'd_cost',
    'n_gUpdates',
    'n_dUpdates',
]

if __name__ == '__main__':
    if globalConfig.dataset == 'ICVL':
        ds = Dataset()
        for l in '-22-5 2014 45 67-5'.split():
            ds.loadICVL(l, tApp=True)

    elif globalConfig.dataset == 'NYU':
        ds = Dataset()
        for i in range(0,75000,20000):
            ds.loadNYU(i, tApp=True)

    elif globalConfig.dataset == 'MSRA':
        pid = 0
        ds = Dataset()
        for i in range(0,9):
            if i == pid:
                continue
            ds.loadMSRA('P%d'%i, tApp=True)

    else:
        raise ValueError('unknown dataset %s'%globalConfig.dataset)

    z_dim = 23

    print 'training length = %d'%len(ds.frmList)
    ds.frmToNp()
    train_stream = DataStream(ds.x_norm)

    gan = DepthGAN(z_dim=z_dim, batch_size=100, lr=0.0003)
    gan.genLossAndGradient()
    gan.train(train_stream, 'dummy', 1001, isRm=True, isFeaMat=True)
