from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, concatenate, Conv2DTranspose
from keras.layers.core import Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, AveragePooling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Deconv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
import datetime
import matplotlib.pyplot as plt
import sys
sys.path.append("../utils")
#from data_loader import DataLoader
import numpy as np
import os
import data_utils
import models
import math
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
import gc
from skimage.measure import compare_ssim, compare_psnr
import generate

import keras.backend as K

def psnr(y_true, y_pred):
    import keras.backend as K
    return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)

class SRGAN():
    def __init__(self, batchsize):
        self.batch_size = batchsize
        # Input shape
        self.lr_channels = 3
        self.lr_height = 512                 # Low resolution height
        self.lr_width = 512                  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.lr_channels)
        self.hr_channels = 3
        self.hr_height = self.lr_height   # High resolution height
        self.hr_width = self.lr_width     # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.hr_channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = 16

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.ggf = 64
        self.df = 64

        # Optimizers used by networks
        optimizer_vgg = Adam(0.0001, 0.9)
        optimizer_generator = Adam(1e-4, 0.9)
        optimizer_discriminator = Adam(0.0001, 0.9)
        optimizer_gan = Adam(0.0001, 0.9)

        # Build basic networks
        self.vgg = self.build_vgg(optimizer_vgg) # model1
        self.discriminator = self.build_discriminator(optimizer_discriminator)
        self.generator = self.build_generator(optimizer_generator)

        # Plot the generator
        #from keras.utils import plot_model
        #plot_model(self.generator, to_file="../../figures/srgan-generator.png", show_shapes=True, show_layer_names=True)
        #self.generator.summary()
        #self.discriminator.summary()

        #Build combined network
        self.combined = self.build_srgan(optimizer_gan)  
        

    def init_pix2pix(self, batch_size, dataset):
        #load raw data
        self.dset = dataset
        image_data_format = "channels_last"
        patch_size = (128,128)
        use_mbd = False
        bn_mode = 2
        self.rows = 4
        self.cols = 4
        img_dim = (512,512,3)
        nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size, image_data_format)

        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = data_utils.load_split_datasets(self.dset, image_data_format)
        weights_epoch = 5
        generator = "deconv"

    def build_vgg(self, optimizer):        
        print("Build VGG")
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py

        img = Input(shape=self.hr_shape)
        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]

        # Extract image features
        img_features = vgg(img)

        # Create model and compile
        model = Model(inputs=img, outputs=img_features)
        model.trainable = False
        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        return model
    
    def build_srgan(self,optimizer):
        print("Build Combined SRGAN")
        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_batch = Input(shape=(4,512,512,3))

        # Generate high res. version from low res.
        fake_hr = self.generator([img_batch])

        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        #model = Model([img_batch, img_hr], [validity, fake_features])
        model = Model([img_batch, img_hr], [validity, fake_hr])
        model.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer,
                              metrics=[psnr])

        return model        

    def build_generator(self, optimizer):
        print("Build Generator")
        channel_axis = -1
        self.init = "glorot_uniform"

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation("relu")(d)
            d = BatchNormalization(momentum=0.8, axis=channel_axis)(d)
            #d = LeakyReLU(alpha=0.25)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8, axis=channel_axis)(d)
            #d = Add()([d, layer_input])
            m = Add()([d, layer_input])
            return m

        def deconv2d(layer_input, filters):
            """Layers used during upsampling"""
            #u = Conv2D(128, kernel_size=3, strides=1, activation='linear', padding='same')(layer_input)
            #u = LeakyReLU(alpha=0.25)(u)
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=3, strides=1, padding='same')(u)
            u = Activation("relu")(u)
            return u

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(axis=channel_axis)(d)
            return d
        
        def conv_block_unet(x, f, name, bn_mode, bn_axis, bn=True, strides=(2,2)):
            x = LeakyReLU(0.2)(x)
            x = Conv2D(f, (3, 3), strides=strides, name=name, padding="same")(x)
            if bn:
                x = BatchNormalization(axis=bn_axis)(x)
            return x

        def deconv_block_unet_lstm(x, x_0_2, x_1_2, x_2_2, x_3_2, f, h, w, batch_size, name, bn_mode, bn_axis, bn=True, dropout=False):
            x = Activation("relu")(x)
            x = Deconv2D(f, (3, 3), strides=(2, 2), padding="same")(x)
            if bn:
                x = BatchNormalization(axis=bn_axis)(x)
            if dropout:
                x = Dropout(0.5)(x)
            x = Concatenate(axis=bn_axis)([x, x_0_2, x_1_2, x_2_2, x_3_2])

            return x

        # Low resolution image input
        img_input = Input(shape=[4,512,512,3])

        # ====================================================
        # ================= GLOBAL Generator ==================
        # ====================================================
        
        img_dim = (128,128,3)
        nb_filters = self.ggf
        bn_axis = -1
        bn_mode = 2
        h, w, nb_channels = img_dim
        min_s = min(img_dim[:-1])

        # Prepare encoder filters
        nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
        list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]
        # Encoder

        lambda_0 = Lambda(lambda x: x[:,0,:,:,:])(img_input)
        lambda_1 = Lambda(lambda x: x[:,1,:,:,:])(img_input)
        lambda_2 = Lambda(lambda x: x[:,2,:,:,:])(img_input)
        lambda_3 = Lambda(lambda x: x[:,3,:,:,:])(img_input)

        downsampled_0 = AveragePooling2D(pool_size=(4,4))(lambda_0)
        downsampled_1 = AveragePooling2D(pool_size=(4,4))(lambda_1)
        downsampled_2 = AveragePooling2D(pool_size=(4,4))(lambda_2)
        downsampled_3 = AveragePooling2D(pool_size=(4,4))(lambda_3)

        list_encoder_0 = [Conv2D(list_nb_filters[0], (3, 3),
                               strides=(2, 2), name="global_unet_conv2D_0_1", padding="same")(downsampled_0)]
        list_encoder_1 = [Conv2D(list_nb_filters[0], (3, 3),
                               strides=(2, 2), name="global_unet_conv2D_1_1", padding="same")(downsampled_1)]
        list_encoder_2 = [Conv2D(list_nb_filters[0], (3, 3),
                               strides=(2, 2), name="global_unet_conv2D_2_1", padding="same")(downsampled_2)]
        list_encoder_3 = [Conv2D(list_nb_filters[0], (3, 3),
                               strides=(2, 2), name="global_unet_conv2D_3_1", padding="same")(downsampled_3)]
        # update current "image" h and w
        h, w = h / 2, w / 2
        for i, f in enumerate(list_nb_filters[1:]):
            name0 = "global_unet_conv2D_0_%s" % (i + 2)
            conv0 = conv_block_unet(list_encoder_0[-1], f, name0, bn_mode, bn_axis)
            list_encoder_0.append(conv0)

            name1 = "global_unet_conv2D_1_%s" % (i + 2)
            conv1 = conv_block_unet(list_encoder_1[-1], f, name1, bn_mode, bn_axis)
            list_encoder_1.append(conv1)

            name2 = "global_unet_conv2D_2_%s" % (i + 2)
            conv2 = conv_block_unet(list_encoder_2[-1], f, name2, bn_mode, bn_axis)
            list_encoder_2.append(conv2)

            name3 = "global_unet_conv2D_3_%s" % (i + 2)
            conv3 = conv_block_unet(list_encoder_3[-1], f, name3, bn_mode, bn_axis)
            list_encoder_3.append(conv3)

            h, w = h / 2, w / 2

        # Combine in LSTM:
        unet_concat = Concatenate(axis=1)([list_encoder_0[-1], list_encoder_1[-1], list_encoder_2[-1], list_encoder_3[-1]])
        unet_reshape = Reshape((4, 512))(unet_concat)
        unet_lstm = LSTM(512, input_shape=(4,512))(unet_reshape)
        unet_reshape_back = Reshape((1, 1, 512))(unet_lstm)

        # Prepare decoder filters
        list_nb_filters = list_nb_filters[:-1][::-1]
        if len(list_nb_filters) < nb_conv - 1:
            list_nb_filters.append(nb_filters)

        # Decoder
        list_decoder = [deconv_block_unet_lstm(unet_reshape_back, list_encoder_0[-2], list_encoder_1[-2], list_encoder_2[-2], list_encoder_3[-2],
                                          list_nb_filters[0], h, w, self.batch_size,
                                          "global_unet_upconv2D_1", bn_mode, bn_axis, dropout=True)]
        h, w = h * 2, w * 2
        for i, f in enumerate(list_nb_filters[1:]):
            name = "global_unet_upconv2D_%s" % (i + 2)
            # Dropout only on first few layers
            if i < 2:
                d = True
            else:
                d = False
            conv = deconv_block_unet_lstm(list_decoder[-1], list_encoder_0[-(i + 3)], list_encoder_1[-(i + 3)], list_encoder_2[-(i + 3)], list_encoder_3[-(i + 3)], f, h,
                                     w, self.batch_size, name, bn_mode, bn_axis, dropout=d)
            list_decoder.append(conv)
            h, w = h * 2, w * 2

        x = Activation("relu")(list_decoder[-1])
        x = Deconv2D(nb_channels, (3, 3), strides=(2, 2), padding="same")(x)
        global_generated = Activation("tanh")(x)
            
        # ====================================================
        # ================= LOCAL Generator =================
        # ====================================================
        
        img_dim = (128,128,3)
        nb_filters = self.ggf
        bn_axis = -1
        bn_mode = 2
        h, w, nb_channels = img_dim
        min_s = min(img_dim[:-1])

        # Prepare encoder filters
        nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
        list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]
        # Encoder

        patch_0_0 = Lambda(lambda x: x[:,:,0:128,0:128,:])(img_input)
        patch_0_1 = Lambda(lambda x: x[:,:,0:128,128:256,:])(img_input)
        patch_0_2 = Lambda(lambda x: x[:,:,0:128,256:384,:])(img_input)
        patch_0_3 = Lambda(lambda x: x[:,:,0:128,384:512,:])(img_input)

        patch_1_0 = Lambda(lambda x: x[:,:,128:256,0:128,:])(img_input)
        patch_1_1 = Lambda(lambda x: x[:,:,128:256,128:256,:])(img_input)
        patch_1_2 = Lambda(lambda x: x[:,:,128:256,256:384,:])(img_input)
        patch_1_3 = Lambda(lambda x: x[:,:,128:256,384:512,:])(img_input)

        patch_2_0 = Lambda(lambda x: x[:,:,256:384,0:128,:])(img_input)
        patch_2_1 = Lambda(lambda x: x[:,:,256:384,128:256,:])(img_input)
        patch_2_2 = Lambda(lambda x: x[:,:,256:384,256:384,:])(img_input)
        patch_2_3 = Lambda(lambda x: x[:,:,256:384,384:512,:])(img_input)

        patch_3_0 = Lambda(lambda x: x[:,:,384:512,0:128,:])(img_input)
        patch_3_1 = Lambda(lambda x: x[:,:,384:512,128:256,:])(img_input)
        patch_3_2 = Lambda(lambda x: x[:,:,384:512,256:384,:])(img_input)
        patch_3_3 = Lambda(lambda x: x[:,:,384:512,384:512,:])(img_input)
        
        local_batch = Concatenate(axis=0)([patch_0_0, patch_0_1, patch_0_2, patch_0_3, patch_1_0, patch_1_1, patch_1_2, patch_1_3, patch_2_0, patch_2_1, patch_2_2, patch_2_3, patch_3_0, patch_3_1, patch_3_2, patch_3_3])

        lambda_0 = Lambda(lambda x: x[:,0,:,:,:])(local_batch)
        lambda_1 = Lambda(lambda x: x[:,1,:,:,:])(local_batch)
        lambda_2 = Lambda(lambda x: x[:,2,:,:,:])(local_batch)
        lambda_3 = Lambda(lambda x: x[:,3,:,:,:])(local_batch)

        list_encoder_0 = [Conv2D(list_nb_filters[0], (3, 3),
                               strides=(2, 2), name="local_unet_conv2D_0_1", padding="same")(lambda_0)]
        list_encoder_1 = [Conv2D(list_nb_filters[0], (3, 3),
                               strides=(2, 2), name="local_unet_conv2D_1_1", padding="same")(lambda_1)]
        list_encoder_2 = [Conv2D(list_nb_filters[0], (3, 3),
                               strides=(2, 2), name="local_unet_conv2D_2_1", padding="same")(lambda_2)]
        list_encoder_3 = [Conv2D(list_nb_filters[0], (3, 3),
                               strides=(2, 2), name="local_unet_conv2D_3_1", padding="same")(lambda_3)]
        # update current "image" h and w
        h, w = h / 2, w / 2
        for i, f in enumerate(list_nb_filters[1:]):
            name0 = "local_unet_conv2D_0_%s" % (i + 2)
            conv0 = conv_block_unet(list_encoder_0[-1], f, name0, bn_mode, bn_axis)
            list_encoder_0.append(conv0)

            name1 = "local_unet_conv2D_1_%s" % (i + 2)
            conv1 = conv_block_unet(list_encoder_1[-1], f, name1, bn_mode, bn_axis)
            list_encoder_1.append(conv1)

            name2 = "local_unet_conv2D_2_%s" % (i + 2)
            conv2 = conv_block_unet(list_encoder_2[-1], f, name2, bn_mode, bn_axis)
            list_encoder_2.append(conv2)

            name3 = "local_unet_conv2D_3_%s" % (i + 2)
            conv3 = conv_block_unet(list_encoder_3[-1], f, name3, bn_mode, bn_axis)
            list_encoder_3.append(conv3)

            h, w = h / 2, w / 2

        # Combine in LSTM:
        unet_concat = Concatenate(axis=1)([list_encoder_0[-1], list_encoder_1[-1], list_encoder_2[-1], list_encoder_3[-1]])
        unet_reshape = Reshape((4, 512))(unet_concat)
        unet_lstm = LSTM(512, input_shape=(4,512))(unet_reshape)
        unet_reshape_back = Reshape((1, 1, 512))(unet_lstm)

        # Prepare decoder filters
        list_nb_filters = list_nb_filters[:-1][::-1]
        if len(list_nb_filters) < nb_conv - 1:
            list_nb_filters.append(nb_filters)

        # Decoder
        list_decoder = [deconv_block_unet_lstm(unet_reshape_back, list_encoder_0[-2], list_encoder_1[-2], list_encoder_2[-2], list_encoder_3[-2],
                                          list_nb_filters[0], h, w, self.batch_size,
                                          "local_unet_upconv2D_1", bn_mode, bn_axis, dropout=True)]
        h, w = h * 2, w * 2
        for i, f in enumerate(list_nb_filters[1:]):
            name = "local_unet_upconv2D_%s" % (i + 2)
            # Dropout only on first few layers
            if i < 2:
                d = True
            else:
                d = False
            conv = deconv_block_unet_lstm(list_decoder[-1], list_encoder_0[-(i + 3)], list_encoder_1[-(i + 3)], list_encoder_2[-(i + 3)], list_encoder_3[-(i + 3)], f, h,
                                     w, self.batch_size, name, bn_mode, bn_axis, dropout=d)
            list_decoder.append(conv)
            h, w = h * 2, w * 2

        x = Activation("relu")(list_decoder[-1])
        x = Deconv2D(nb_channels, (3, 3), strides=(2, 2), padding="same")(x)
        local_generated = Activation("tanh")(x)

        patch_0_0 = Lambda(lambda x: x[0*self.batch_size:1*self.batch_size,:,:,:])(local_generated)
        patch_0_1 = Lambda(lambda x: x[1*self.batch_size:2*self.batch_size,:,:,:])(local_generated)
        patch_0_2 = Lambda(lambda x: x[2*self.batch_size:3*self.batch_size,:,:,:])(local_generated)
        patch_0_3 = Lambda(lambda x: x[3*self.batch_size:4*self.batch_size,:,:,:])(local_generated)

        patch_1_0 = Lambda(lambda x: x[4*self.batch_size:5*self.batch_size,:,:,:])(local_generated)
        patch_1_1 = Lambda(lambda x: x[5*self.batch_size:6*self.batch_size,:,:,:])(local_generated)
        patch_1_2 = Lambda(lambda x: x[6*self.batch_size:7*self.batch_size,:,:,:])(local_generated)
        patch_1_3 = Lambda(lambda x: x[7*self.batch_size:8*self.batch_size,:,:,:])(local_generated)

        patch_2_0 = Lambda(lambda x: x[8*self.batch_size:9*self.batch_size,:,:,:])(local_generated)
        patch_2_1 = Lambda(lambda x: x[9*self.batch_size:10*self.batch_size,:,:,:])(local_generated)
        patch_2_2 = Lambda(lambda x: x[10*self.batch_size:11*self.batch_size,:,:,:])(local_generated)
        patch_2_3 = Lambda(lambda x: x[11*self.batch_size:12*self.batch_size,:,:,:])(local_generated)

        patch_3_0 = Lambda(lambda x: x[12*self.batch_size:13*self.batch_size,:,:,:])(local_generated)
        patch_3_1 = Lambda(lambda x: x[13*self.batch_size:14*self.batch_size,:,:,:])(local_generated)
        patch_3_2 = Lambda(lambda x: x[14*self.batch_size:15*self.batch_size,:,:,:])(local_generated)
        patch_3_3 = Lambda(lambda x: x[15*self.batch_size:16*self.batch_size,:,:,:])(local_generated)

        local_generated_row_0 = Concatenate(axis=2)([patch_0_0, patch_0_1, patch_0_2, patch_0_3])
        local_generated_row_1 = Concatenate(axis=2)([patch_1_0, patch_1_1, patch_1_2, patch_1_3])
        local_generated_row_2 = Concatenate(axis=2)([patch_2_0, patch_2_1, patch_2_2, patch_2_3])
        local_generated_row_3 = Concatenate(axis=2)([patch_3_0, patch_3_1, patch_3_2, patch_3_3])

        local_generated = Concatenate(axis=1)([local_generated_row_0, local_generated_row_1, local_generated_row_2, local_generated_row_3])

        # ====================================================
        # ===================== SR GAN =======================
        # ====================================================

        #d1 = d_block(local_generated, self.gf, strides=2)
        #d2 = d_block(d1, self.gf, strides=2)        
        d2 = AveragePooling2D(pool_size=(4,4))(local_generated)
    

        conc = concatenate([global_generated, d2])

        # Pre-residual block
        c1 = Conv2D(self.gf, kernel_size=9, strides=1, padding='same')(conc)
        c1 = Activation('relu')(c1)
        #c1 = BatchNormalization(axis=channel_axis)(c1);
        #c1 = LeakyReLU(alpha=0.2)(c1);

        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = Conv2D(self.gf, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8, axis=channel_axis)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2, self.gf)
        u2 = deconv2d(u1, self.gf)

        # Generate high resolution output
        gen_hr = Conv2D(self.hr_channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        model = Model(inputs=img_input, outputs=gen_hr)
        model.compile(
            loss='mse',
            optimizer=optimizer
        )

        return model

    def build_discriminator(self, optimizer):
        print("Build Discriminator")

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)

        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        model = Model(inputs=d0, outputs=validity)
        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        return model

    def generate_train_batch(self, batch_size):
        # generate batch from pix2pix generators 
        Y_data, X_data = next(data_utils.gen_batch(self.Y_train, self.X_train, batch_size))
        return Y_data, X_data

    def train(self, epochs, batch_size=1, sample_interval=0):        
        print("Start Training")

        start_time = datetime.datetime.now()

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_batch = self.generate_train_batch(batch_size) #self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict([imgs_batch])

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_batch = self.generate_train_batch(batch_size) #self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_batch, imgs_hr], [valid, imgs_hr])

            elapsed_time = datetime.datetime.now() - start_time            
            
            generated_images_val = []

            # Scores on validation set:
            count = 0
            psnr_score = 0
            ssim_score = 0
            for k in range(int(math.ceil(np.shape(self.Y_val)[0] / self.batch_size))):
                count += self.batch_size
                Y_val_batch, X_val_batch = next(data_utils.gen_batch(self.Y_val, self.X_val, self.batch_size))
                g_val = self.generator.predict(X_val_batch)
                generated_images_val.insert(0, g_val[0])
                generated_images_val.insert(0, g_val[1])
                for i in range(self.batch_size):
                    ssim_score += compare_ssim(Y_val_batch[i], g_val[i], multichannel=True)
                    psnr_score += compare_psnr(Y_val_batch[i], g_val[i])
            psnr_score = psnr_score / count
            ssim_score = ssim_score / count


            count = 0
            psnr_score_test = 0
            ssim_score_test = 0
            for k in range(int(math.ceil(np.shape(self.Y_val)[0] / self.batch_size))):
                count += self.batch_size
                Y_test_batch, X_test_batch = next(data_utils.gen_batch(self.Y_test, self.X_test, self.batch_size))
                g_val = self.generator.predict(X_test_batch)
                #print("Gval: " + str(np.shape(g_val)))
                generated_images_val.insert(0, g_val[0])
                generated_images_val.insert(0, g_val[1])
                for i in range(self.batch_size):
                    ssim_score_test += compare_ssim(Y_test_batch[i], g_val[i], multichannel=True)
                    psnr_score_test += compare_psnr(Y_test_batch[i], g_val[i])
            psnr_score_test = psnr_score_test / count
            ssim_score_test = ssim_score_test / count

            val_metric_names = ["PSNR", "SSIM"]
            val_metric = [psnr_score, ssim_score]
            val_metric_test = [psnr_score_test, ssim_score_test]

            print("Epoch {}/{} | Time: {}s\n>> Generator: {}\n>> Discriminator: {}\n>> Validation Set: {}\n>> Test Set: {}\n".format(
                    epoch, epochs,
                    (datetime.datetime.now() - start_time).seconds,
                    ", ".join(["{}={:.3e}".format(k, v) for k, v in zip(self.combined.metrics_names, g_loss)]),
                    ", ".join(["{}={:.3e}".format(k, v) for k, v in zip(self.discriminator.metrics_names, d_loss)]),
                    ", ".join(["{}={:.3e}".format(k, v) for k, v in zip(val_metric_names, val_metric)]),
                    ", ".join(["{}={:.3e}".format(k, v) for k, v in zip(val_metric_names, val_metric_test)])
            ))

            # If at save interval => save generated image samples
            self.sample_images(epoch)
            self.sample_images(epoch, "val")
            self.sample_images(epoch, "test")
    
            if epoch % 25 == 0:
                print("Saving Weights")
                generate.generate_video(self.generator, 25, False)
                gen_weights_path = os.path.join('weights/gen_weights_epoch_%s.h5' % (epoch))
                self.generator.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join('weights/disc_weights_epoch_%s.h5' % (epoch))
                self.discriminator.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join('weights/combined_weights_epoch_%s.h5' % (epoch))
                self.combined.save_weights(DCGAN_weights_path, overwrite=True) 
                

    def sample_images(self, epoch, plotSet="train"):
        r, c = 2, 2
        
        if plotSet == "train":            
            imgs_hr, imgs_batch = next(data_utils.gen_batch(self.Y_train, self.X_train, self.batch_size)) 
        elif plotSet == "val":
            imgs_hr, imgs_batch = next(data_utils.gen_batch(self.Y_val, self.X_val, self.batch_size))
        else:
            imgs_hr, imgs_batch = next(data_utils.gen_batch(self.Y_test, self.X_test, self.batch_size))
        fake_hr = self.generator.predict([imgs_batch])

        imgs_lr_local = imgs_batch[:,:,:,0:3]

        # Rescale images 0 - 1
        imgs_lr_local = 0.5 * imgs_lr_local + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        if plotSet == "train":            
            fig.savefig("images/train/train_%d.png" % (epoch))
        elif plotSet == "val":
            fig.savefig("images/val/val_%d.png" % (epoch))
        else:
            fig.savefig("images/test/test_%d.png" % (epoch))
        plt.close()

        # Save low resolution images for comparison
        #for i in range(r):
        #    fig = plt.figure()
        #    plt.imshow(imgs_lr_local[i])
        #    fig.savefig('images/%d_lowres%d.png' % (epoch, i))
        #    plt.close()

if __name__ == '__main__':   

    parser = argparse.ArgumentParser(description='SRGan model')
    parser.add_argument('--dset', type=str, default="flocking_512", help="Dataset Name")
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--nepoch', default=100, type=int, help="Number of epochs")

    #data_utils.generate_split_hd5("flocking_512", "channels_last")
    batchsize = args.batch_size
    gan = SRGAN(batchsize)
    gan.init_pix2pix(batchsize, args.dset)
    gan.train(epochs=args.nepoch, batch_size=batchsize, sample_interval=50)
