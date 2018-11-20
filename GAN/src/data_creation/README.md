directory structure of datasets

├── dataset
    ├── train 
    ├── test 
    ├── val



#Build Dataset:

`python make_dataset.py`

positional arguments:

    jpeg_dir             path to jpeg images (name of dataset directory)
    nb_channels          number of image channels

optional arguments:

    -h, --help           show this help message and exit
    --img_size IMG_SIZE  Desired Width == Height
    --do_plot            Plot the images to make sure the data processing went
                         OK



**Example:**

`python make_dataset.py flocking_512 3 --img_size 512 --do_plot True`

#Split into train / test / val sets:

Use the `generate_split_hd5` function from split_h5_in_train_test_val.py

arguments:

    dset                    name of dataset directory
    load                    array that determines the generated datasets, default = ['train', 'test', 'val']                            
    size                    image size, default = 512
    nb_channels             number of channels, default = 3
