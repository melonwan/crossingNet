import theano, theano.tensor as T, lasagne
from numpy.random import RandomState
import numpy.linalg
import cv2, time, shutil, os, json
from collections import namedtuple

import globalConfig
from forwardRender import ForwardRender
from data.dataset import *
from data.util import Camera
from data.evaluation import Evaluation
from data.stream import DataStream, MultiDataStream
import scipy.optimize

class GanRender(ForwardRender):
    DisErr = namedtuple('disErr', ['gan', 'est', 'metric'])
    GenErr = namedtuple('genErr', ['gan', 'recons', 'metric'])
    golden_max = 1.0 
    
    def __init__(self, x_dim, rndGanInput=False, metricCombi=False):
        super(GanRender, self).__init__(x_dim)
        self.rndGanInput = rndGanInput
        self.metricCombi = metricCombi

    def genLossAndGradient(self, isTrain=True):
        self.depth_gan.linkSubNets()
        # similar to GAN, need both discriminative and generative

        # the generative part
        gen_params = self.alignment_params +\
                self.depth_gan.gen_params
        fake_render_var = lasagne.layers.get_output(
            self.depth_gan.dis_render_layer,
            self.render_var,
            deterministic=False)
        real_render_var = lasagne.layers.get_output(
            self.depth_gan.dis_render_layer,
            self.depth_gan.depth_input_var,
            deterministic=False)
        # recons_loss = abs(real_render_var - fake_render_var)
        recons_loss = (real_render_var - fake_render_var)**2
        recons_loss = T.clip(recons_loss, 0, self.golden_max)
        recons_loss = T.mean(recons_loss)

        # gan part 
        real_feamat_var = lasagne.layers.get_output(self.depth_gan.feamat_layer, 
                                                    self.depth_gan.depth_input_var,
                                                    deterministic=False)
        real_feamat_var = real_feamat_var.mean(axis=0)
        combi_weights_input_var = T.fmatrix('noise_combination')
        latent_noises_var = T.dot(combi_weights_input_var, self.latent_var)

        aligned_gan_noise_var = lasagne.layers.get_output(self.alignment_layer,
                                                          latent_noises_var,
                                                          deterministic=False)
        fake_depth_var = lasagne.layers.get_output(self.render_layer,
                                                   aligned_gan_noise_var,
                                                   deterministic=False)
        gan_fake_depth_var = T.concatenate([fake_depth_var, self.render_var],
                                       axis=0)

        px_fake_var = lasagne.layers.get_output(self.depth_gan.dis_px_layer,
                                                gan_fake_depth_var,
                                                deterministic=False)
        px_real_var = lasagne.layers.get_output(self.depth_gan.dis_px_layer,
                                                self.depth_gan.depth_input_var,
                                                deterministic=False)

        loss_dis_fake = lasagne.objectives.binary_crossentropy(px_fake_var,
                                                               T.zeros(px_fake_var.shape))
        loss_dis_fake = lasagne.objectives.aggregate(loss_dis_fake, mode='mean')
        loss_dis_real = lasagne.objectives.binary_crossentropy(px_real_var,
                                                               T.ones(px_real_var.shape))
        loss_dis_real = lasagne.objectives.aggregate(loss_dis_real, mode='mean')
        loss_dis_gan = loss_dis_real + loss_dis_fake
        fake_feamat_var = lasagne.layers.get_output(self.depth_gan.feamat_layer,
                                                    gan_fake_depth_var,
                                                    deterministic=False)
        fake_feamat_var = fake_feamat_var.mean(axis=0)
        gan_loss_gen = T.mean(abs(real_feamat_var - fake_feamat_var)) 
        gan_loss_gen = lasagne.objectives.aggregate(gan_loss_gen, mode='mean')

        # metric part 
        if not self.metricCombi:
            self.metric_layer = self.depth_gan.build_metric(output_dim = self.z_dim)
            fake_metric_var = lasagne.layers.get_output(
                self.metric_layer,
                fake_depth_var,
                deterministic=False)
            real_metric_var = lasagne.layers.get_output(
                self.metric_layer,
                self.depth_gan.depth_input_var,
                deterministic=False)
            self_metric_var = lasagne.layers.get_output(
                self.metric_layer,
                self.render_var,
                deterministic=False
            )

            latent_diff = self.latent_var - latent_noises_var
            metric_diff = real_metric_var - fake_metric_var
            self_diff = real_metric_var - self_metric_var

        else:
            self.metric_layer, self.metric_combilayer = \
                self.depth_gan.build_metric_combi(output_dim = self.z_dim)
            fake_metric_var = lasagne.layers.get_output(
                self.metric_layer,
                fake_depth_var,
                deterministic=False)
            real_metric_var = lasagne.layers.get_output(
                self.metric_layer,
                self.depth_gan.depth_input_var,
                deterministic=False)
            self_metric_var = lasagne.layers.get_output(
                self.metric_layer,
                self.render_var,
                deterministic=False
            )

            latent_diff = self.latent_var - latent_noises_var
            real_fake_combi_var = T.concatenate([real_metric_var, fake_metric_var], 
                                                axis=1)
            metric_diff = lasagne.layers.get_output(self.metric_combilayer,
                                                    real_fake_combi_var)
            self_combi_var = T.concatenate([real_metric_var, self_metric_var],
                                           axis=1)
            self_diff = lasagne.layers.get_output(self.metric_combilayer,
                                                  self_combi_var)
            

        metric_loss = (latent_diff - metric_diff)**2 + self_diff**2
        metric_loss = metric_loss.mean()

        gen_loss = gan_loss_gen + recons_loss + metric_loss
        gen_update_var = lasagne.updates.adam(gen_loss,
                                             gen_params,
                                             self.lr,
                                             self.b1)

        if self.rndGanInput:
            gan_train_fn_input = [
                 self.pose_vae.pose_input_var,
                 self.origin_input_var,
                 self.pose_vae.noise_input_var,
                 combi_weights_input_var,
                 self.depth_gan.depth_input_var
            ]
        else:
            gan_train_fn_input = [
                 self.pose_vae.pose_input_var,
                 self.origin_input_var,
                 self.pose_vae.noise_input_var,
                 self.depth_gan.depth_input_var
            ]
        gen_train_fn_output = [
            gan_loss_gen,
            recons_loss,
            metric_loss
        ]
        
        if isTrain:
            self.gen_train_fn = theano.function(
                gan_train_fn_input,
                gen_train_fn_output,
                updates = gen_update_var)
            print 'gen_train_fn compiled'

        # alignment part
        align_update_var = lasagne.updates.adam(recons_loss,
                                                self.alignment_params,
                                                self.lr*10.,
                                                self.b1)
        if isTrain:
            self.alignment_train_fn = theano.function(
                [
                 self.pose_vae.pose_input_var,
                 self.origin_input_var,
                 self.pose_vae.noise_input_var,
                 self.depth_gan.depth_input_var
                ],
                recons_loss,
                updates = align_update_var)
            print 'alignment_train_fn compiled'


        # estimating the latent variable part
        z_est_layer = self.depth_gan.build_recognition(self.z_dim)
        z_est_var = lasagne.layers.get_output(z_est_layer, 
                                             self.depth_gan.depth_input_var,
                                             deterministic=False)
        z_est_tvar = lasagne.layers.get_output(z_est_layer,
                                              self.depth_gan.depth_input_var,
                                              deterministic=True)
        loss_dis_est = (z_est_var - self.latent_var)**2
        # loss_dis_est = loss_dis_est.sum(axis=1)
        loss_dis_est = lasagne.objectives.aggregate(loss_dis_est,
                                                    mode='mean')

        dis_loss = loss_dis_gan\
                   + loss_dis_est\
                   + metric_loss
        dis_update_var = lasagne.updates.adam(dis_loss,
                                              self.depth_gan.dis_params+\
                                              self.depth_gan.reco_params+\
                                              self.depth_gan.metric_params,
                                              self.lr,
                                              self.b1)
        if self.rndGanInput:
            dis_train_fn_input = [
                 self.pose_vae.pose_input_var,
                 self.origin_input_var,
                 self.pose_vae.noise_input_var,
                 combi_weights_input_var,
                 self.depth_gan.depth_input_var
            ]
        else:
            dis_train_fn_input = [
                 self.pose_vae.pose_input_var,
                 self.origin_input_var,
                 self.pose_vae.noise_input_var,
                 self.depth_gan.depth_input_var
            ]
        dis_train_fn_output = [
            loss_dis_gan,
            loss_dis_est,
            metric_loss
        ]
        if isTrain:
            self.dis_train_fn = theano.function(
                dis_train_fn_input,
                dis_train_fn_output,
                updates = dis_update_var)
            print 'dis_train_fn compiled' 

        # initialize the training of recognition, metric part
        init_dis_loss = loss_dis_est + metric_loss
        init_dis_update_var = lasagne.updates.adam(init_dis_loss,
                                                  self.depth_gan.reco_params+\
                                                  self.depth_gan.metric_params,
                                                  self.lr*10.,
                                                  self.b1)
        if self.rndGanInput:
            init_dis_train_fn_input = [
                 self.pose_vae.pose_input_var,
                 self.origin_input_var,
                 self.pose_vae.noise_input_var,
                 combi_weights_input_var,
                 self.depth_gan.depth_input_var
            ]
        else:
            init_dis_train_fn_input = [
                 self.pose_vae.pose_input_var,
                 self.origin_input_var,
                 self.pose_vae.noise_input_var,
                 self.depth_gan.depth_input_var
            ]
        init_dis_train_fn_output = [
            loss_dis_est,
            metric_loss
        ]
        if isTrain:
            self.init_dis_train_fn = theano.function(
                init_dis_train_fn_input,
                init_dis_train_fn_output,
                updates = init_dis_update_var)
            print 'init_dis_fn compiled'

        # rendering function 
        self.render_fn = theano.function(
            [self.pose_vae.pose_input_var,
             self.origin_input_var,
             self.pose_vae.noise_input_var],
            self.render_tvar
        )
        print 'rendering function compiled'

        # estimation function
        self.z_est_fn = theano.function(
            [self.depth_gan.depth_input_var],
            z_est_tvar
        )
        print 'z_est function compiled'

        # pose reconstruction function
        self.pose_vae.genLossAndGradient()
        self.vae_reco_fn = self.pose_vae.recons_fn
        print 'pose vae function compiled'

        # pose decoding function
        est_pose_z_var = T.fmatrix('est_pose_z')
        est_pose_tvar = lasagne.layers.get_output(self.pose_vae.recons_layer,
                                                 est_pose_z_var,
                                                 deterministic=True)
        self.pose_decode_fn = theano.function(inputs=[est_pose_z_var],
                                             outputs=est_pose_tvar)
        self.pose_encode_fn = self.pose_vae.encode_fn
        print 'pose decoder fnction compiled'


    def train(self, nepoch, train_stream, val_stream=None, desc='dummy'):
        cache_dir = os.path.join(globalConfig.model_dir, 'gan_render/%s_%s'%(globalConfig.dataset,desc))

        model_dir = os.path.join(cache_dir, 'pretrained_model')
        cache_dir = os.path.join(cache_dir, desc)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        log_path = os.path.join(cache_dir, 'log.txt')
        flog = open(log_path, 'w')
        flog.close()

        img_dir = os.path.join(cache_dir, 'img')
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        os.mkdir(img_dir)

        param_dir = os.path.join(cache_dir, 'params')
        if not os.path.exists(param_dir):
            os.mkdir(param_dir)
        
        self.pose_vae.loadParam(globalConfig.vae_pretrain_path)
        self.depth_gan.loadParam(globalConfig.gan_pretrain_path)

        seed = 42
        np_rng = RandomState(seed)
        print '[ganRender] begin training loop'
        for epoch in range(0, nepoch):
            gen_errs, dis_errs = np.zeros((3,)), np.zeros((3,))
            gen_err, dis_err, nupdates = 0, 0, 0 
            start_time = time.time()
            for pose, orig, trans, com, depth in\
                train_stream.iterate(batchsize=self.batch_size, shuffle=True):
                vae_noises = np_rng.normal(0, 0.05,
                                      (self.batch_size, self.pose_z_dim)
                                      ).astype(np.float32)
                gan_noises = GanRender.rndCvxCombination(np_rng,
                                                         src_num=self.batch_size,
                                                         tar_num=self.batch_size,
                                                         sel_num=5)

                if epoch < 21:
                    gen_err = self.alignment_train_fn(pose,orig,vae_noises,depth)
                    gen_errs += np.array([0.,1.,0.])*gen_err
                    est_err, metr_err =\
                        self.init_dis_train_fn(pose,orig,vae_noises,gan_noises,depth)
                    dis_errs += np.array([0.,1.,0.])*est_err +\
                        np.array([0.,0.,1.])*metr_err
                    nupdates += 1
                    continue
                if self.rndGanInput:
                    dis_errs +=\
                        np.asarray(self.dis_train_fn(pose,orig,vae_noises,gan_noises,depth))
                    gen_errs += \
                        np.asarray(self.gen_train_fn(pose,orig,vae_noises,gan_noises,depth))
                else:
                    dis_errs += \
                        np.asarray(self.dis_train_fn(pose,orig,vae_noises,depth))
                    gen_errs += \
                        np.asarray(self.gen_train_fn(pose,orig,vae_noises,depth))
                nupdates += 1

            print 'Epoch {} for {} took {:.3f}s'.format(epoch, nepoch, time.time()-start_time)
            dis_errs /= nupdates
            gen_errs /= nupdates
            dis_errs = self.DisErr(*tuple(dis_errs))
            gen_errs = self.GenErr(*tuple(gen_errs))
            print 'disErr: {}'.format(dis_errs)
            print 'genErr: {}'.format(gen_errs)
            flog = open(log_path, 'a')
            flog.write('epoch {}, {}s\n'.format(epoch, time.time()-start_time))
            flog.write(json.dumps((dis_errs, gen_errs))+'\n')
            flog.close()

            if epoch % 10 == 0 and val_stream is not None:
                idx = 0
                for skel, orig, trans, com, depth in\
                    val_stream.iterate(batchsize=1, shuffle=False):
                        noise = np.zeros((1,self.pose_z_dim), np.float32)
                        reco_depth = self.render_fn(skel, orig, noise) 
                        reco_pose = self.vae_reco_fn(skel, noise)
                        pose = self.resumePose(reco_pose[0], 
                                               orig[0])
                        fake_img = self.visPair(reco_depth[0],
                                                pose,
                                                trans[0],
                                                com[0], 50.0)

                        pose = self.resumePose(skel[0],
                                               orig[0])
                        real_img = self.visPair(depth[0], 
                                                pose,
                                                trans[0],
                                                com[0], 50.0)

                        est_z = self.z_est_fn(depth)
                        est_z.shape = (23,)
                        est_z, est_orig = est_z[:20], est_z[20:]
                        est_z.shape = (1,20)
                        est_orig.shape = (1,3)
                        est_pose = self.pose_decode_fn(est_z)
                        est_depth = self.render_fn(est_pose, est_orig, noise)
                        pose = self.resumePose(est_pose[0],
                                               est_orig[0])
                        est_img = self.visPair(est_depth[0],
                                               pose,
                                               trans[0],
                                               com[0], 50.0)
                        com_img = self.visPair(depth[0],
                                               pose,
                                               trans[0],
                                               com[0], 50.0)

                        recons_depth = np.hstack((real_img, fake_img, est_img, com_img))
                        cv2.imwrite(os.path.join(img_dir,'%d_%d.jpg'%(epoch,idx)),\
                                    recons_depth.astype('uint8'))
                        idx += 1
            
            if epoch % 10 == 0:
                self.saveParam(os.path.join(param_dir, '-1'))

            if epoch % 100 == 0:
                self.saveParam(os.path.join(param_dir, '%d'%epoch))
        flog.close()

    def compileEnergyFn(self):
        # the energy function and its derivatives are used later for 
        # optimization for testing
        given_x_var = T.ftensor4('given_x_var')
        est_z_var = T.fmatrix('init_z_var')
        
        align_var = lasagne.layers.get_output(self.alignment_layer,
                                              est_z_var,
                                              deterministic=True)
        est_x_var = lasagne.layers.get_output(self.render_layer,
                                              inputs=align_var,
                                              deterministic=True)
        test_pixel_loss = (given_x_var - est_x_var)**2
        test_pixel_loss = T.clip(test_pixel_loss, 0, self.golden_max)
        test_pixel_loss = lasagne.objectives.aggregate(
            test_pixel_loss, mode='mean')

        est_feat_var = lasagne.layers.get_output(self.metric_layer,
                                                inputs=est_x_var,
                                                deterministic=True)
        giv_feat_var = lasagne.layers.get_output(self.metric_layer,
                                                inputs=given_x_var,
                                                deterministic=True)
        if self.metricCombi:
            metric_combi_var = T.concatenate([giv_feat_var,est_feat_var],
                                            axis=1)
            learned_update = lasagne.layers.get_output(self.metric_combilayer,
                                                      metric_combi_var)
        else:
            learned_update = giv_feat_var - est_feat_var 
        
        self_loss = est_z_var**2
        self_loss = T.mean(self_loss)

        learned_update *= -1
        test_metric_loss = learned_update**2
        test_metric_loss = T.mean(test_metric_loss)
        test_loss = test_metric_loss + 0.001*self_loss
        self.energy_fn = theano.function(
            [est_z_var, given_x_var],
            test_loss,
            allow_input_downcast=True
        )
        print 'self.energy_fn compiled'
        
        def gradFn(dz_var, est_z_var, given_x_var):
            align_var = lasagne.layers.get_output(self.alignment_layer,
                                                  est_z_var,
                                                  deterministic=True)
            est_x_var = lasagne.layers.get_output(self.render_layer,
                                                  inputs=align_var,
                                                  deterministic=True)
            test_pixel_loss = (given_x_var - est_x_var)**2
            test_pixel_loss = T.clip(test_pixel_loss, 0, self.golden_max)
            test_pixel_loss = lasagne.objectives.aggregate(
                test_pixel_loss, mode='mean')

            est_feat_var = lasagne.layers.get_output(self.metric_layer,
                                                    inputs=est_x_var,
                                                    deterministic=True)
            giv_feat_var = lasagne.layers.get_output(self.metric_layer,
                                                    inputs=given_x_var,
                                                    deterministic=True)
            if self.metricCombi:
                metric_combi_var = T.concatenate([giv_feat_var,est_feat_var],
                                                axis=1)
                learned_update = lasagne.layers.get_output(self.metric_combilayer,
                                                          metric_combi_var)
            else:
                learned_update = giv_feat_var - est_feat_var 

            learned_update *= -1
            test_metric_loss = learned_update**2
            test_metric_loss = T.mean(test_metric_loss)

            self_loss = est_z_var**2
            self_loss = T.mean(self_loss)

            # test_loss = test_pixel_loss
            test_loss = test_metric_loss + 0.001*self_loss

            # test_loss = test_pixel_loss
            dz_var = theano.grad(test_loss, est_z_var)
            return dz_var 

        grad_z_var = T.fmatrix('grad_z')
        grad, updates = theano.scan(
            fn=gradFn,
            outputs_info=[T.as_tensor_variable(np.zeros((1,self.z_dim),np.float32))],
            non_sequences=[est_z_var, given_x_var],
            n_steps=T.as_tensor_variable(1)
        )
        grad = grad[-1]
        print 'scan loop calculated'
        
        self.energy_grad_fn = theano.function(inputs=[est_z_var, given_x_var], 
                                         outputs=grad[0],
                                         updates=updates,
                                         allow_input_downcast=True)
        print 'energy_grad_fn compiled'

    def bfgs(self, est_z, depth, maxiter=20):
        new_z = scipy.optimize.fmin_bfgs(
            f=lambda x:self.energy_fn(x.reshape(1,-1),depth),
            x0=est_z,
            fprime=lambda x:self.energy_grad_fn(x.reshape(1,-1),depth),
            # maxiter=maxiter,
            full_output=False,
            disp=False
        )
        return new_z
        
    def genTestEstGrad(self):
        def updateFn(est_z_var, given_x_var):
            # output of render image given est_z_var
            align_var = lasagne.layers.get_output(self.alignment_layer,
                                                  est_z_var,
                                                  deterministic=True)
            est_x_var = lasagne.layers.get_output(self.render_layer,
                                                  inputs=align_var,
                                                  deterministic=True)
            # test_pixel_loss = abs(given_x_var - est_x_var)
            test_pixel_loss = (given_x_var - est_x_var)**2
            # test_pixel_loss = T.clip(test_pixel_loss, 0, self.golden_max)
            test_pixel_loss = lasagne.objectives.aggregate(
                test_pixel_loss, mode='mean')

            est_feat_var = lasagne.layers.get_output(self.metric_layer,
                                                    inputs=est_x_var,
                                                    deterministic=True)
            giv_feat_var = lasagne.layers.get_output(self.metric_layer,
                                                    inputs=given_x_var,
                                                    deterministic=True)
            if self.metricCombi:
                metric_combi_var = T.concatenate([giv_feat_var,est_feat_var],
                                                axis=1)
                learned_update = lasagne.layers.get_output(self.metric_combilayer,
                                                          metric_combi_var)
            else:
                learned_update = giv_feat_var - est_feat_var 

            learned_update *= -1

            test_metric_loss = learned_update**2
            test_metric_loss = T.mean(test_metric_loss)

            # test_loss = test_pixel_loss + test_metric_loss*0.1
            test_loss = test_metric_loss
            # test_loss = test_pixel_loss

            # dz_var = theano.grad(test_loss, est_z_var)
            dz_var = learned_update 
            return est_z_var-0.1*dz_var

        # our initial estimation
        est_z_var = T.fmatrix('est_z')
        # given image during test time
        given_x_var = T.ftensor4('given_x')
        # number of updates
        K = T.iscalar('K')
        new_z, updates = theano.scan(
            fn=updateFn,
            outputs_info=[est_z_var],
            non_sequences=[given_x_var],
            n_steps=K
        )
        new_z = new_z[-1]
        print 'scan loop calculated'
        
        self.search_fn = theano.function(inputs=[est_z_var, given_x_var, K], 
                                         outputs=new_z,
                                         updates=updates)
        print 'scan gradient function compiled'

    def test(self, test_stream, desc = 'dummy', modelIdx = '-1'):
        cache_dir = os.path.join(globalConfig.model_dir, 'gan_render/%s_%s'%(globalConfig.dataset,desc))

        if not os.path.exists(cache_dir):
            raise IOError('%s does not exists'%cache_dir)
         
        img_dir = os.path.join(cache_dir, 'test_img')
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        os.mkdir(img_dir)

        model_path = os.path.join(cache_dir, 'params', modelIdx)
        self.loadParam(model_path)

        idx, total_time, recons_err = 0, 0, 0
        maxJntError = []
        
        codec = cv2.cv.CV_FOURCC('X','V','I','D')
        vid = cv2.VideoWriter(os.path.join(img_dir,'res.avi'), codec, 25, (128*4,128))
        for skel, orig, trans, com, depth in\
            test_stream.iterate(batchsize=1, shuffle=False):
                noise = np.zeros((1,self.pose_z_dim), np.float32)
                reco_depth = self.render_fn(skel, orig, noise) 
                reco_pose = self.vae_reco_fn(skel, noise)
                pose = self.resumePose(reco_pose[0], 
                                       orig[0])
                fake_img = self.visPair(reco_depth[0],
                                        pose,
                                        trans[0],
                                        com[0], 50.0)

                gt_pose = self.resumePose(skel[0],
                                       orig[0])
                real_img = self.visPair(depth[0], 
                                        gt_pose,
                                        trans[0],
                                        com[0], 50.0)

                # real calculation part
                start_time = time.time()
                est_z = self.z_est_fn(depth)
                est_z.shape = (23,)
                est_z, est_orig = est_z[:20], est_z[20:]
                est_z.shape = (1,20)
                est_orig.shape = (1,3)
                est_pose = self.pose_decode_fn(est_z)
                end_time = time.time()

                est_depth = self.render_fn(est_pose, est_orig, noise)
                est_pose = self.resumePose(est_pose[0],
                                       est_orig[0])
                est_img = self.visPair(est_depth[0],
                                       est_pose,
                                       trans[0],
                                       com[0], 50.0)
                com_img = self.visPair(depth[0],
                                       est_pose,
                                       trans[0],
                                       com[0], 50.0)

                recons_err += (abs(reco_depth-depth)).mean()

                recons_depth = np.hstack((real_img, fake_img, est_img, com_img))
                # cv2.imwrite(os.path.join(img_dir,'%d_0.jpg'%(idx)),\
                            # recons_depth.astype('uint8'))
                idx += 1
                maxJntError.append(Evaluation.maxJntError(gt_pose, est_pose))
                total_time += end_time - start_time
                vid.write(recons_depth.astype('uint8'))
        
        print 'average running time = %fs'%(total_time/idx)
        print 'average reconstruction error  = %f'%(recons_err/idx)
        fig_path = os.path.join(img_dir, 'maxError.txt')
        Evaluation.plotError(maxJntError,fig_path)
        vid.release()

    def prepareData(self, ds):
        ds.normTranslation()
        ds.frmToNp()

        data_depth = ds.x_norm
        data_pose_skel = ds.y_norm # skeleton in conanical view

        ndata = len(ds.frmList)

        data_pose_orig = np.zeros((ndata, 3), np.float32)
        data_pose_trans = np.zeros((ndata, 3, 3), np.float32)
        data_pose_com = np.zeros((ndata, 3), np.float32)
        for i, frm in enumerate(ds.frmList):
            data_pose_orig[i] = frm.origin
            data_pose_trans[i] = frm.trans
            data_pose_com[i] = frm.com3D
        print '[ganRender] data prepared with %d samples'%ndata
        print '[ganRender] x_norm range: {} to {}'.format(data_depth.min(),
                                                          data_depth.max())
        print '[ganRender] origin range: {} to {}'.format(data_pose_orig.min(),
                                                          data_pose_orig.max())

        return MultiDataStream([data_pose_skel,
                               data_pose_orig,
                               data_pose_trans,
                               data_pose_com,
                               data_depth])
    @classmethod
    def rndCvxCombination(cls, rng, src_num, tar_num, sel_num):
        # generate tar_num random convex combinations from src_num point, every
        # time only sel_num from the src_num are used
        if sel_num > src_num:
            raise ValueError('sel_num %d should less then src_num %d'%(sel_num,
                                                                      src_num))
        m = np.zeros((tar_num, src_num))
        for s in m:
            sel = rng.choice(src_num, sel_num)
            s[sel] = rng.uniform(0,1,(sel_num,)) 
            s /= s.sum()
        return m.astype(np.float32)


    def saveParam(self, path):
        self.saveAlignParam(path)
        self.pose_vae.saveParam(path)
        self.depth_gan.saveParam(path)
        print 'parameters has been saved to %s'%path

    def loadParam(self, path):     
        self.loadAlignParam(path)
        self.pose_vae.loadParam(path)
        self.depth_gan.loadParam(path)
        print 'parameters has been loaded from %s'%path


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
        raise ValueError('unkonwn dataset %s'%globalConfig.dataset)

    print 'validation length = %d'%len(val_ds.frmList)
    skel_num = len(val_ds.frmList[0].norm_skel)
    print 'skel_num=%d'%skel_num
    render = GanRender(skel_num, rndGanInput=True, metricCombi=False)
    train_stream = render.prepareData(ds)
    val_stream = render.prepareData(val_ds)

    desc = 'pretrained'
    render.genLossAndGradient(isTrain=True)
    render.train(101,train_stream, val_stream,\
                 desc=desc)
    render.test(val_stream, desc=desc, modelIdx='-1')

