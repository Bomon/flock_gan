# Training and evaluating

## Training

`python srgan.py`

optional arguments:

    -h, --help            show this help message and exit
    --dset DSET           dataset name
    --batch_size BATCH_SIZE
                        Batch size
    --nepoch EPOCH         Number of epochs


**Example for flocking_512:**

`python srgan.py --dset "flocking_512" --batch_size 2 --nepoch 300`


### Expected outputs:

- Weights are saved in weights
- Figures are saved in ../../figures
- Save model weights every few epochs
- Image samples during training are saved in ./images


## Generate

`python generate.py`

The video sequence to predict is located in ./vid/[0-9]*
