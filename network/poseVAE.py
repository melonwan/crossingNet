import lasagne, lasagne.utils
import numpy.linalg
import data.dataset
import theano, theano.tensor as T
import shutil, math
from data.dataset import *
import time, cv2, sys, cPickle
import sklearn.decomposition
from numpy.random import RandomState
from data.util import vis_pose
import globalConfig
CreateParam = lasagne.utils.create_param

init_weight = lasagne.init.Normal(mean=0., std=0.01)
init_biase = lasagne.init.Constant(0.)
init_gamma = lasagne.init.Normal(mean=1, std=0.01)

class PoseVAE(object):
    def __init__(self, x_dim, pca_dim = 40, z_dim = 20, batch_size = 200, lr = 0.0003, b1 = 0.5):
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.pca_dim = pca_dim
        
        self.batch_size = 200
        self.lr = lr
        self.b1 = b1

        self.noise_input_var = T.fmatrix('noise')
        self.noise_input_layer = lasagne.layers.InputLayer(shape=(None, self.z_dim),
                                                   input_var=self.noise_input_var)

        self.pose_input_var = T.fmatrix('input')
        self.pose_input_layer = lasagne.layers.InputLayer(shape=(None,self.x_dim),
                                                        input_var=self.pose_input_var)
        

        # init the network
        self.kl_loss_layer, self.mu_layer, self.logsd_layer =\
                self.build_encoder(self.pose_input_layer)
        self.z_layer =\
                self.build_sampler(self.mu_layer, 
                                   self.logsd_layer, 
                                   self.noise_input_layer)
        self.z_var = lasagne.layers.get_output(self.z_layer,
                                               deterministic=False)
        self.z_tvar = lasagne.layers.get_output(self.z_layer,
                                                deterministic=True)
        self.recons_layer = self.build_decoder()

        # used for training, batch_norm parameters depend only on batch input
        self.recons_var = lasagne.layers.get_output(self.recons_layer,
                                                    self.z_var,
                                                    deterministic=False)
        # used for testing, batch_norm parameters is previsouly average
        self.recons_tvar = lasagne.layers.get_output(self.recons_layer,
                                                     self.z_tvar,
                                                     deterministic=True)
    
    def gen_pca(self, data):
        pca = sklearn.decomposition.PCA(n_components=self.pca_dim, whiten=False)
        pca.fit(data)
        self.pca_weights = pca.components_

    def savePcaParam(self, path):
        f = open(path, 'wb')
        cPickle.dump((self.scale, self.pca), f,
                     protocol=cPickle.HIGHEST_PROTOCOL)
        print 'pca parameters has been saved to %s'%path

    def loadPcaParam(self, path):
        f = open(path, 'rb')
        self.scale, self.pca = cPickle.load(f)
        print 'pca parameters has been successfully loaded %s'%path

    def build_encoder(self, input_layer):
        # init parameter
        nw1 = self.pca_dim 
        ew_1 = CreateParam(init_weight, (self.x_dim, nw1), 'ew_1')
        self.ew = ew_1
        eb_1 = CreateParam(init_biase, (nw1,), 'eb_1')
        eg_1 = CreateParam(init_gamma, (nw1,), 'eg_1')

        w_mu = CreateParam(init_weight, (nw1, self.z_dim), 'w_mu')
        b_mu = CreateParam(init_biase, (self.z_dim,), 'b_mu')

        w_logsd = CreateParam(init_weight,(nw1, self.z_dim), 'w_logsd')
        b_logsd = CreateParam(init_biase, (self.z_dim,), 'b_logsd')

        dense_1 =  lasagne.layers.DenseLayer(input_layer,
                                      nonlinearity=lasagne.nonlinearities.rectify,
                                      num_units = nw1, 
                                      W=ew_1, 
                                      b=eb_1)
        
        # mu layer
        mu_layer = lasagne.layers.DenseLayer(dense_1,
                                           nonlinearity=lasagne.nonlinearities.identity,
                                           num_units=self.z_dim,
                                           W = w_mu,
                                           b = b_mu)
        meansq_layer = lasagne.layers.ExpressionLayer(mu_layer,
                                                 T.sqr,
                                                 output_shape='auto')

        # variance layer
        logsd_layer = lasagne.layers.DenseLayer(dense_1,
                                               nonlinearity=lasagne.nonlinearities.identity,
                                               num_units=self.z_dim,
                                               W = w_logsd,
                                               b = b_logsd)
        sd_layer = lasagne.layers.ExpressionLayer(logsd_layer,
                                                  T.exp,
                                                  output_shape='auto')

        kl_div_layer = lasagne.layers.ElemwiseSumLayer([meansq_layer,sd_layer,logsd_layer],
                                                       [0.5,0.5,-0.5])
        kl_div_layer = lasagne.layers.ExpressionLayer(kl_div_layer,
                                                      lambda x: x+np.float32(-0.5),
                                                     output_shape='auto')
        kl_loss_layer = lasagne.layers.ExpressionLayer(kl_div_layer,
                                                       lambda x: T.sum(x, axis=1),
                                                      output_shape='auto')

        self.encoder_params = [ew_1, eb_1,
                               w_mu, b_mu,
                               w_logsd, b_logsd]
        nPara = len(self.encoder_params)

        self.encoder_all_params = \
                lasagne.layers.get_all_params(kl_loss_layer)[-nPara:]

        return kl_loss_layer, mu_layer, logsd_layer

    # reparameterization part
    def build_sampler(self, mu_layer, logsd_layer, noise_input_layer):
        sigma_layer = lasagne.layers.ExpressionLayer(logsd_layer,
                                                         lambda x: T.exp(0.5*x),
                                                         output_shape='auto')

        noise_layer = lasagne.layers.ElemwiseMergeLayer(
            [sigma_layer, noise_input_layer], T.mul)

        z_layer = lasagne.layers.ElemwiseSumLayer(
            [mu_layer, noise_layer],
            [1, 1])

        return z_layer

    def build_decoder(self):
        nw1 = self.pca_dim 
        dw_1 = CreateParam(init_weight, (self.z_dim, nw1), 'dw_1')
        db_1 = CreateParam(init_biase, (nw1,), 'db_1')
        dg_1 = CreateParam(init_gamma, (nw1,), 'dg_1')

        nw_mu = self.x_dim
        dw_mu = CreateParam(init_weight, (nw1, nw_mu), 'dw_mu')
        db_mu = CreateParam(init_biase, (nw_mu,), 'db_mu')
        self.dw = dw_mu

        input_layer = lasagne.layers.InputLayer(shape=(None, self.z_dim))
        dense_1 = lasagne.layers.DenseLayer(input_layer,
                                      nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                      num_units = nw1, 
                                      W=dw_1, 
                                      b=db_1)

        dense_recons = lasagne.layers.DenseLayer(dense_1,
                                           nonlinearity=lasagne.nonlinearities.identity,
                                           num_units=nw_mu,
                                           W = dw_mu,
                                           b = db_mu)

        self.decoder_params = [dw_1, db_1,
                               dw_mu, db_mu]
        nPara = len(self.decoder_params)
        self.decoder_all_params =\
                lasagne.layers.get_all_params(dense_recons)[-nPara:]
        
        return dense_recons

    def genLossAndGradient(self):
        # establish loss
        kl_div = lasagne.layers.get_output(self.kl_loss_layer,
                                           deterministic=False)
        kl_loss = lasagne.objectives.aggregate(kl_div, mode='sum')

        # assume the reconstructed all with standard Gaussian distribution
        recons_loss = lasagne.objectives.squared_error(self.recons_var,
                                                       self.pose_input_var)
        recons_loss = recons_loss*0.5
        recons_loss = lasagne.objectives.aggregate(recons_loss, mode='sum')
        
        # calculate gradient
        loss = kl_loss + recons_loss
        # loss = recons_loss
        lr_var = T.fscalar('lr')
        update_params = self.encoder_params + self.decoder_params
        update_vars = lasagne.updates.adam(loss, update_params, 
                                           lr_var, self.b1)

        # compile the function
        self.train_fn = theano.function(
            [self.pose_input_var, self.noise_input_var, lr_var],
            loss,
            updates = update_vars)
        self.recons_fn = theano.function(
            [self.pose_input_var, self.noise_input_var],
            self.recons_tvar
        )
        self.encode_fn = theano.function(
            [self.pose_input_var, self.noise_input_var],
            self.z_tvar
        )
        
        print '[PoseVAE]function compiled'

    def saveParam(self, path):
        encoder_para_val = [param.get_value() for param in self.encoder_all_params]
        decoder_para_val = [param.get_value() for param in self.decoder_all_params]

        encoder_path = path+'_en_params.npz'
        np.savez(encoder_path, *encoder_para_val)

        decoder_path = path+'_de_params.npz'
        np.savez(decoder_path, *decoder_para_val)
        print '[PoseVAE] parameter has been saved to %s_en/de_params.npz'%path

    def loadParam(self, path):
        encoder_path = path+'_en_params.npz'
        en = np.load(encoder_path)
        for idx, param in enumerate(self.encoder_all_params):
            param.set_value(en['arr_%d'%idx])

        decoder_path = path+'_de_params.npz'
        de = np.load(decoder_path)
        for idx, param in enumerate(self.decoder_all_params):
            param.set_value(de['arr_%d'%idx])
        print '[PoseVAE] paramter has been loaded from %s_en/de_params.npz'%path

    def train(self, train_stream, val_stream=None, nepoch=0, desc = 'dummy'):
        print '[PoseVAE] enter the training loop with %d epoches'%nepoch
        
        cache_dir = os.path.join(globalConfig.model_dir, 'pose_vae/%s_%s'%(globalConfig.dataset,desc))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        img_dir = '%s/img/'%cache_dir
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        os.mkdir(img_dir)

        para_dir = '%s/params/'%cache_dir
        if not os.path.exists(para_dir):
            os.mkdir(para_dir)

        seed = 42
        np_rng = RandomState(seed)

        if hasattr(self, 'pca_weights'):
            print 'initialize the weights from PCA weights'
            self.ew.set_value(self.pca_weights.transpose())
            self.dw.set_value(self.pca_weights)
        
        curr_lr = self.lr
        for epoch in range(nepoch):
            tr_err, nupdates = 0, 0
            start_time = time.time()
            for pose in train_stream.iterate(batchsize=self.batch_size,
                                                      shuffle=True):
                noises = np_rng.normal(0, 0.05, 
                                       (self.batch_size, self.z_dim)).astype(np.float32)
                tr_err += self.train_fn(pose, noises, curr_lr)
                nupdates += 1

            curr_lr = self.lr*math.pow(0.95, epoch*(nupdates+1)/10000.0)

            if val_stream is not None and epoch%20 == 0:
                noises = np.zeros((1, self.z_dim), np.float32)
                val_err, nval = 0, 0
                for val_pose in val_stream.iterate(batchsize=1,
                                                   shuffle=False):
                    recons_res = self.recons_fn(val_pose, noises)
                    val_err += ((val_pose - recons_res)**2).sum()
                    nval += 1
                    init_res = val_pose.copy()
                    if hasattr(self, 'pca'):
                        recons_res *= self.scale
                        recons_res = self.pca.inverse_transform(recons_res)
                        init_res *= self.scale
                        init_res = self.pca.inverse_transform(init_res)

                    for recons_pose, init_pose in zip(recons_res, init_res):
                        recons_skel = vis_pose(recons_pose)
                        init_skel = vis_pose(init_pose)
                        img = np.hstack((init_skel, recons_skel))
                        cv2.imwrite('%s/%d_%d.jpg'%(img_dir,epoch,nval),
                                   img)

                print 'validation error = {}'.format(val_err/nval)

            if epoch%50 == 0:
                self.saveParam('%s/%d'%(para_dir, -1))

            if epoch%100 == 0:
                self.saveParam('%s/%d'%(para_dir, epoch))

            print 'Epoch {} of {} took {:.3f}s'.format(epoch, nepoch,
                                                      time.time()-start_time)
            print 'loss = {}'.format(tr_err/nupdates)

from data.stream import DataStream
if __name__ == '__main__':
    if globalConfig.dataset == 'ICVL':
        ds = Dataset()
        for l in '-22-5 2014 45 67-5'.split():
            ds.loadICVL(l, tApp=True)

        val_ds = Dataset()
        val_ds.loadICVLTest()

    elif globalConfig.dataset == 'NYU':
        ds = Dataset()
        for i in range(0,75000,20000):
            ds.loadNYU(i, tApp=True)

        val_ds = Dataset()
        val_ds.loadNYU(0, tFlag='test')

    elif globalConfig.dataset == 'MSRA':
        pid = 0
        ds = Dataset()
        for i in range(0,9):
            if i == pid:
                continue
            ds.loadMSRA('P%d'%i, tApp=True)

        val_ds = Dataset()
        val_ds.loadMSRA('P%d'%pid)

    else:
        raise ValueError('unknown dataset %s'%globalConfig.dataset)

    print 'loaded over with %d samples'%len(ds.frmList)
    ds.normTranslation()
    ds.frmToNp()
    ds.frmList = None
    ds.x_norm = None
    
    skel_dim = len(ds.y_norm[0])
    print 'skeleton dim=%d'%skel_dim
    pca_dim = 40
    vae = PoseVAE(x_dim = skel_dim)
    vae.genLossAndGradient()
    vae.gen_pca(ds.y_norm)

    val_ds.normTranslation()
    val_ds.frmToNp()
    
    desc = 'dummy'
    data_len = len(ds.y_norm)
    train_stream = DataStream(ds.y_norm)
    val_stream = DataStream(val_ds.y_norm)
    vae.train(train_stream, val_stream, 201, desc)
    
