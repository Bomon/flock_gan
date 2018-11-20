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
import srgan
    

def inverse_normalization(X):

    return (X + 1.) / 2.

def normalization(X):

    return X / 127.5 - 1

def generate_video(generator_model, frames, load_weights=True):
    print("load data")
    # Load and rescale data

    try:
        print("load model")  

        if load_weights:
            gen_weights_path = os.path.join('./weights/gen_weights_epoch_200.h5')
            generator_model.load_weights(gen_weights_path)
        
        
        folderlist = sorted(glob.glob('./vid/[0-9-]*'))
        print(folderlist)
        for folder in folderlist:        
            print("Generate Animation: " + str(folder))
            filelist = sorted(glob.glob(folder + '/*.jpg'))
            images = np.array([normalization(np.array(Image.open(fname))) for fname in filelist])

            for i in range(frames):
                frame01 = images[i]            
                frame02 = images[i+1]
                frame03 = images[i+2]
                frame04 = images[i+3]

                frame_batch = np.zeros((2,4,512,512,3))
                frame_batch[0,0,:,:,:] = frame01
                frame_batch[0,1,:,:,:] = frame02
                frame_batch[0,2,:,:,:] = frame03
                frame_batch[0,3,:,:,:] = frame04

                for j in range(4):
                    
                    plt.imshow(inverse_normalization(frame_batch[:,j,:,:,:])[0])
                    plt.axis("off")
                    plt.clf()
                    plt.close()

                # Generate images
                X_gen = generator_model.predict(frame_batch)[0]
                images = np.vstack([images, np.expand_dims(X_gen,0)])
                X_gen = inverse_normalization(X_gen)[0]

                plt.imshow(X_gen)
                plt.axis("off")
                if i == 0:
                    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d_%H:%M:%S')
                    plt.savefig("./vid/predicted/prediction_" + str(filter(str.isdigit, folder)) + '_' + str(datestring) +".jpg")
                plt.clf()
                plt.close()

            datestring = datetime.strftime(datetime.now(), '%Y-%m-%d_%H:%M:%S')
            imageio.mimsave('./vid/predicted/animation_'+ str(filter(str.isdigit, folder)) + '_' + str(datestring) +'.gif', images)        

    except KeyboardInterrupt:
        pass          

if __name__ == "__main__":    
    batchsize = 2
    gan = srgan.SRGAN(batchsize)

    generator = gan.generator
    
    generate_video(generator, 25)
