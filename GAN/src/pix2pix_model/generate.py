import os
import argparse
import sys
import time
import numpy as np
import models
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import keras.backend as K
import glob
import matplotlib.pylab as plt
import Image
from matplotlib.pyplot import figure
# Utils
sys.path.append("../utils")
import general_utils
import data_utils
import imageio
from datetime import datetime
    

def inverse_normalization(X):

    return (X + 1.) / 2.

def normalization(X):

    return X / 127.5 - 1

def generate_video(**kwargs):

    # Roll out the parameters
    generator = kwargs["generator"]
    image_data_format = "channels_last"
    img_dim = kwargs["img_dim"]
    patch_size = kwargs["patch_size"]
    bn_mode = kwargs["bn_mode"]
    dset = kwargs["dset"]
    weights_epoch = kwargs["weights_epoch"]
    use_mbd = kwargs["use_mbd"]
    frames = kwargs["frames"]
    batch_size = 1

    print("load data")
    # Load and rescale data
    X_target_train, X_frames_train, X_target_val, X_frames_val = data_utils.load_data(dset, image_data_format)
    img_dim = X_target_train.shape[-3:]

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size, image_data_format)

    try:
        print("load model")        

        # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load generator model
        generator_model = models.load("generator_unet_%s" % generator,
                                      img_dim,
                                      nb_patch,
                                      bn_mode,
                                      use_mbd,
                                      batch_size)

        generator_model.compile(loss='mae', optimizer=opt_discriminator)

        gen_weights_path = os.path.join('weights/gen_weights_epoch%s.h5' % (weights_epoch))
        generator_model.load_weights(gen_weights_path)
        
        filelist = sorted(glob.glob('generate/*.jpg'))
        images = np.array([normalization(np.array(Image.open(fname))) for fname in filelist])
        print(filelist)

        for i in range(frames):
            print("Generate Frame " + str(i))

            frame01 = images[i]            
            frame02 = images[i+1]
            frame03 = images[i+2]
            frame04 = images[i+3]

            #print(np.shape(frame01))
            frame_batch = np.zeros((1,4,128,128,3))
            frame_batch[:,0,:,:,:] = frame01
            frame_batch[:,1,:,:,:] = frame02
            frame_batch[:,2,:,:,:] = frame03
            frame_batch[:,3,:,:,:] = frame04

            for j in range(4):
                
                plt.imshow(inverse_normalization(frame_batch[:,j,:,:,:])[0])
                plt.axis("off")
                plt.savefig("generate/predicted/"+ "{:02}".format(i+j) +".jpg")
                plt.clf()
                plt.close()

            # Generate images
            X_gen = generator_model.predict(frame_batch)
            images = np.vstack([images, X_gen])
            #print(np.shape(images))
            X_gen = inverse_normalization(X_gen)[0]

            plt.imshow(X_gen)
            plt.axis("off")
            plt.savefig("generate/predicted/"+ "{:02}".format(i+9) +".jpg")
            plt.clf()
            plt.close()

        datestring = datetime.strftime(datetime.now(), '%Y-%m-%d_%H:%M:%S')
        imageio.mimsave('generate/animation/animation_'+ str(datestring) +'.gif', images)        

    except KeyboardInterrupt:
        pass          

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('patch_size', type=int, nargs=2, action="store", help="Patch size for D")
    parser.add_argument('--backend', type=str, default="theano", help="theano or tensorflow")
    parser.add_argument('--generator', type=str, default="upsampling", help="upsampling or deconv")
    parser.add_argument('--dset', type=str, default="facades", help="facades")
    parser.add_argument('--bn_mode', default=2, type=int, help="Batch norm mode")
    parser.add_argument('--img_dim', default=64, type=int, help="Image width == height")
    parser.add_argument('--use_mbd', action="store_true", help="Whether to use minibatch discrimination")
    parser.add_argument('--weights_epoch', default="100", type=str, help="Epoch of weights to load")
    parser.add_argument('--frames', default=10, type=int, help="Number of frames to generate")

    args = parser.parse_args()

    # Set the backend by modifying the env variable
    if args.backend == "theano":
        os.environ["KERAS_BACKEND"] = "theano"
    elif args.backend == "tensorflow":
        os.environ["KERAS_BACKEND"] = "tensorflow"

    # Import the backend
    import keras.backend as K

    # manually set dim ordering otherwise it is not changed
    if args.backend == "theano":
        image_data_format = "channels_first"
        K.set_image_data_format(image_data_format)
    elif args.backend == "tensorflow":
        image_data_format = "channels_last"
        K.set_image_data_format(image_data_format)

    import generate

    # Set default params
    d_params = {"dset": args.dset,
                "generator": args.generator,
                "bn_mode": args.bn_mode,
                "img_dim": args.img_dim,
                "patch_size": args.patch_size,
                "use_mbd": args.use_mbd,
                "weights_epoch": args.weights_epoch,
                "frames": args.frames
                }

    # Launch training
    print("Init done")
    generate_video(**d_params)
