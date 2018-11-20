# Training and evaluating

## Training

`python main.py`


positional arguments:
    
    patch_size            Patch size for D

optional arguments:

    -h, --help            show this help message and exit
    --backend BACKEND     theano or tensorflow
    --generator GENERATOR
                        upsampling or deconv
    --dset DSET           facades
    --batch_size BATCH_SIZE
                        Batch size
    --n_batch_per_epoch N_BATCH_PER_EPOCH
                        Number of training epochs
    --nb_epoch NB_EPOCH   Number of batches per epoch
    --epoch EPOCH         Epoch at which weights were saved for evaluation
    --nb_classes NB_CLASSES
                        Number of classes
    --do_plot             Debugging plot
    --bn_mode BN_MODE     Batch norm mode
    --img_dim IMG_DIM     Image width == height
    --use_mbd             Whether to use minibatch discrimination
    --use_label_smoothing
                        Whether to smooth the positive labels when training D
    --label_flipping LABEL_FLIPPING
                        Probability (0 to 1.) to flip the labels when training
                        D


**Example for flocking_512:**

`python main.py 64 64 --generator "deconv" --dset "flocking_512" --backend "tensorflow"`


### Expected outputs:

- Weights are saved in weights
- Figures are saved in ../../figures
- Save model weights every few epochs


## Generate

`python generate.py`


positional arguments:
    
    patch_size            Patch size for D

optional arguments:

    -h, --help            show this help message and exit
    --generator GENERATOR
                        upsampling or deconv
    --dset DSET           facades
    --bn_mode BN_MODE     Batch norm mode
    --img_dim IMG_DIM     Image width == height
    --use_mbd             Whether to use minibatch discrimination
    --weights_epoch       Epoch of weights to load
    --frames              Number of frames to predict



**Example for flocking_512:**

`python generate.py 64 64 --generator "deconv" --dset "flocking_512" --frames 10 --weights_epoch 200`