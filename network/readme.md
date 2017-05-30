# Network
contains all the network implemented via Theano and Lasagne

## depthGAN.py
train the generative adversarial network(GAN) of depth map

## poseVAE.py
train the variational autoencoder(VAE) of pose

## forwardRender.py
to link the two generative models above through an alignment layer

## ganRender.py
to train the posterior based on the pretrained poseVAE and depthGAN
