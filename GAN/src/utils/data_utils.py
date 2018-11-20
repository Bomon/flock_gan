from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import h5py
import cv2
import datetime
import os
import matplotlib.pylab as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split


def normalization(X):

    return X / 127.5 - 1


def inverse_normalization(X):

    return (X + 1.) / 2.


def get_nb_patch(img_dim, patch_size, image_data_format):

    assert image_data_format in ["channels_first", "channels_last"], "Bad image_data_format"

    if image_data_format == "channels_first":
        assert img_dim[1] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[2] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] // patch_size[1])
        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])

    elif image_data_format == "channels_last":
        assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
        img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])

    return nb_patch, img_dim_disc


def extract_patches(X, image_data_format, patch_size):

    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        X = X.transpose(0,2,3,1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)

    return list_X


def load_data(dset, image_data_format):

    with h5py.File("../../data/processed/%s_data.h5" % dset, "r") as hf:

        X_target_train = hf["train_data_frame_target"][:].astype(np.float32)
        X_target_train = normalization(X_target_train)
        X_01_train = hf["train_data_frame_01"][:].astype(np.float32)
        X_01_train = normalization(X_01_train)
        X_02_train = hf["train_data_frame_02"][:].astype(np.float32)
        X_02_train = normalization(X_02_train)
        X_03_train = hf["train_data_frame_03"][:].astype(np.float32)
        X_03_train = normalization(X_03_train)  
        X_04_train = hf["train_data_frame_04"][:].astype(np.float32)
        X_04_train = normalization(X_04_train)  

        if image_data_format == "channels_last":
            X_target_train = X_target_train.transpose(0, 2, 3, 1)
            X_01_train = X_01_train.transpose(0, 2, 3, 1)
            X_02_train = X_02_train.transpose(0, 2, 3, 1)
            X_03_train = X_03_train.transpose(0, 2, 3, 1)
            X_04_train = X_04_train.transpose(0, 2, 3, 1)

        X_target_val = hf["val_data_frame_target"][:].astype(np.float32)
        X_target_val = normalization(X_target_val)
        X_01_val = hf["val_data_frame_01"][:].astype(np.float32)
        X_01_val = normalization(X_01_val)
        X_02_val = hf["val_data_frame_02"][:].astype(np.float32)
        X_02_val = normalization(X_02_val)
        X_03_val = hf["val_data_frame_03"][:].astype(np.float32)
        X_03_val = normalization(X_03_val)  
        X_04_val = hf["val_data_frame_04"][:].astype(np.float32)
        X_04_val = normalization(X_04_val)  

        if image_data_format == "channels_last":
            X_target_val = X_target_val.transpose(0, 2, 3, 1)
            X_01_val = X_01_val.transpose(0, 2, 3, 1)
            X_02_val = X_02_val.transpose(0, 2, 3, 1)
            X_03_val = X_03_val.transpose(0, 2, 3, 1)
            X_04_val = X_04_val.transpose(0, 2, 3, 1)
        
        return X_target_train, [X_01_train, X_02_train, X_03_train, X_04_train], X_target_val, [X_01_val, X_02_val, X_03_val, X_04_val]

def load_data_raw(dset, image_data_format):

    with h5py.File("../../data/processed/%s_data.h5" % dset, "r") as hf:

        X_target_train = hf["train_data_frame_target"][:].astype(np.float32)
        X_target_train = normalization(X_target_train)
        X_01_train = hf["train_data_frame_01"][:].astype(np.float32)
        X_01_train = normalization(X_01_train)
        X_02_train = hf["train_data_frame_02"][:].astype(np.float32)
        X_02_train = normalization(X_02_train)
        X_03_train = hf["train_data_frame_03"][:].astype(np.float32)
        X_03_train = normalization(X_03_train)  
        X_04_train = hf["train_data_frame_04"][:].astype(np.float32)
        X_04_train = normalization(X_04_train)  

        if image_data_format == "channels_last":
            X_target_train = X_target_train.transpose(0, 2, 3, 1)
            X_01_train = X_01_train.transpose(0, 2, 3, 1)
            X_02_train = X_02_train.transpose(0, 2, 3, 1)
            X_03_train = X_03_train.transpose(0, 2, 3, 1)
            X_04_train = X_04_train.transpose(0, 2, 3, 1)
        
        return X_target_train, [X_01_train, X_02_train, X_03_train, X_04_train]

def load_split_datasets(dset, image_data_format, load=["train", "val", "test"]):
    
    start_time = datetime.datetime.now()
    with h5py.File("../../data/processed/%s_data_split.h5" % dset, "r") as hf:
        print("Load Dataset " + str(dset))
        num_instances = len(hf["train_data_frame_target"]) + len(hf["test_data_frame_target"]) + len(hf["val_data_frame_target"])
        print("Num Instances " + str(num_instances))

        X_train = np.array([])
        X_val = np.array([])
        X_test = np.array([])
        Y_train = np.array([])
        Y_val = np.array([])
        Y_test = np.array([])

        if "train" in load:
            print("Load Train")
            Y_train = hf["train_data_frame_target"][:].astype(np.float32)
            Y_train = normalization(Y_train)
            X_01_train = hf["train_data_frame_01"][:].astype(np.float32)
            X_01_train = normalization(X_01_train)
            X_02_train = hf["train_data_frame_02"][:].astype(np.float32)
            X_02_train = normalization(X_02_train)
            X_03_train = hf["train_data_frame_03"][:].astype(np.float32)
            X_03_train = normalization(X_03_train)  
            X_04_train = hf["train_data_frame_04"][:].astype(np.float32)
            X_04_train = normalization(X_04_train)
            print("Size train: " + str(np.shape(Y_train)[0]))
        
        if "val" in load:
            print("Load Val")
            Y_val = hf["val_data_frame_target"][:].astype(np.float32)
            Y_val = normalization(Y_val)
            X_01_val = hf["val_data_frame_01"][:].astype(np.float32)
            X_01_val = normalization(X_01_val)
            X_02_val = hf["val_data_frame_02"][:].astype(np.float32)
            X_02_val = normalization(X_02_val)
            X_03_val = hf["val_data_frame_03"][:].astype(np.float32)
            X_03_val = normalization(X_03_val)  
            X_04_val = hf["val_data_frame_04"][:].astype(np.float32)
            X_04_val = normalization(X_04_val)    
            print("Size val: " + str(np.shape(Y_val)[0]))

        if "test" in load:
            print("Load Test")
            Y_test = hf["test_data_frame_target"][:].astype(np.float32)
            Y_test = normalization(Y_test)
            X_01_test = hf["test_data_frame_01"][:].astype(np.float32)
            X_01_test = normalization(X_01_test)
            X_02_test = hf["test_data_frame_02"][:].astype(np.float32)
            X_02_test = normalization(X_02_test)
            X_03_test = hf["test_data_frame_03"][:].astype(np.float32)
            X_03_test = normalization(X_03_test)  
            X_04_test = hf["test_data_frame_04"][:].astype(np.float32)
            X_04_test = normalization(X_04_test)    
            print("Size test: " + str(np.shape(Y_test)[0]))

        elapsed_time = datetime.datetime.now() - start_time  
        print("Time to load datasets: %s" % (elapsed_time))        


        return [X_01_train, X_02_train, X_03_train, X_04_train], Y_train, [X_01_val, X_02_val, X_03_val, X_04_val], Y_val, [X_01_test, X_02_test, X_03_test, X_04_test], Y_test

def generate_split_hd5(dset, image_data_format, load=["train", "val", "test"], size = 512, nb_channels = 3):    
    
    start_time = datetime.datetime.now()
    with h5py.File("../../data/processed/%s_data.h5" % dset, "r") as hf:
        print("Load Dataset " + str(dset))
        num_instances = len(hf["train_data_frame_target"])
        print("Num Instances " + str(num_instances))

        indices = np.random.permutation(num_instances)
        training_idx, val_idx, test_idx = indices[:int(num_instances*0.7)], indices[int(num_instances*0.7):int(num_instances*0.85)], indices[int(num_instances*0.85):]

        training_idx = sorted(training_idx.tolist())
        val_idx = sorted(val_idx.tolist())
        test_idx = sorted(test_idx.tolist())

        if "train" in load:
            print("Load Train")
            Y_train = hf["train_data_frame_target"][training_idx].astype(np.uint8)
            X_01_train = hf["train_data_frame_01"][training_idx].astype(np.uint8)
            X_02_train = hf["train_data_frame_02"][training_idx].astype(np.uint8)
            X_03_train = hf["train_data_frame_03"][training_idx].astype(np.uint8)
            X_04_train = hf["train_data_frame_04"][training_idx].astype(np.uint8)
            if image_data_format == "channels_last":
                Y_train = Y_train.transpose(0, 2, 3, 1)
                X_01_train = X_01_train.transpose(0, 2, 3, 1)
                X_02_train = X_02_train.transpose(0, 2, 3, 1)
                X_03_train = X_03_train.transpose(0, 2, 3, 1)
                X_04_train = X_04_train.transpose(0, 2, 3, 1)
            print("Size train: " + str(np.shape(Y_train)[0]))
        
        if "val" in load:
            print("Load Val")
            Y_val = hf["train_data_frame_target"][val_idx].astype(np.uint8)
            X_01_val = hf["train_data_frame_01"][val_idx].astype(np.uint8)
            X_02_val = hf["train_data_frame_02"][val_idx].astype(np.uint8)
            X_03_val = hf["train_data_frame_03"][val_idx].astype(np.uint8)
            X_04_val = hf["train_data_frame_04"][val_idx].astype(np.uint8)
            if image_data_format == "channels_last":
                Y_val = Y_val.transpose(0, 2, 3, 1)
                X_01_val = X_01_val.transpose(0, 2, 3, 1)
                X_02_val = X_02_val.transpose(0, 2, 3, 1)
                X_03_val = X_03_val.transpose(0, 2, 3, 1)
                X_04_val = X_04_val.transpose(0, 2, 3, 1)
            print("Size val: " + str(np.shape(Y_val)[0]))

        if "test" in load:
            print("Load Test")
            Y_test = hf["train_data_frame_target"][test_idx].astype(np.uint8)
            X_01_test = hf["train_data_frame_01"][test_idx].astype(np.uint8)
            X_02_test = hf["train_data_frame_02"][test_idx].astype(np.uint8)
            X_03_test = hf["train_data_frame_03"][test_idx].astype(np.uint8)
            X_04_test = hf["train_data_frame_04"][test_idx].astype(np.uint8)
            if image_data_format == "channels_last":
                Y_test = Y_test.transpose(0, 2, 3, 1)
                X_01_test = X_01_test.transpose(0, 2, 3, 1)
                X_02_test = X_02_test.transpose(0, 2, 3, 1)
                X_03_test = X_03_test.transpose(0, 2, 3, 1)
                X_04_test = X_04_test.transpose(0, 2, 3, 1)
            print("Size test: " + str(np.shape(Y_test)[0]))
    
    data_dir = "../../data/processed"
    hdf5_file = os.path.join(data_dir, "%s_data_split.h5" % dset)
    with h5py.File(hdf5_file, "w") as hfw:

        for dset_type in ["train", "test", "val"]:
            print("Build DSet " + str(dset_type))
            data_frame_target = hfw.create_dataset("%s_data_frame_target" % dset_type,
                                             (0, size, size, nb_channels),
                                             maxshape=(None, size, size, 3),
                                             dtype=np.uint8)

            data_frame_01 = hfw.create_dataset("%s_data_frame_01" % dset_type,
                                             (0, size, size, nb_channels),
                                             maxshape=(None, size, size, 3),
                                             dtype=np.uint8)

            data_frame_02 = hfw.create_dataset("%s_data_frame_02" % dset_type,
                                             (0, size, size, nb_channels),
                                             maxshape=(None, size, size, 3),
                                             dtype=np.uint8)

            data_frame_03 = hfw.create_dataset("%s_data_frame_03" % dset_type,
                                             (0, size, size, nb_channels),
                                             maxshape=(None, size, size, 3),
                                             dtype=np.uint8)

            data_frame_04 = hfw.create_dataset("%s_data_frame_04" % dset_type,
                                             (0, size, size, nb_channels),
                                             maxshape=(None, size, size, 3),
                                             dtype=np.uint8)
            if dset_type == "train":
                data_frame_target.resize(np.shape(Y_train)[0], axis=0)
                data_frame_01.resize(np.shape(Y_train)[0], axis=0)
                data_frame_02.resize(np.shape(Y_train)[0], axis=0)
                data_frame_03.resize(np.shape(Y_train)[0], axis=0)
                data_frame_04.resize(np.shape(Y_train)[0], axis=0)
                data_frame_target[:] = Y_train
                data_frame_01[:] = X_01_train
                data_frame_02[:] = X_02_train
                data_frame_03[:] = X_03_train
                data_frame_04[:] = X_04_train

            elif dset_type == "val":
                data_frame_target.resize(np.shape(Y_val)[0], axis=0)
                data_frame_01.resize(np.shape(Y_val)[0], axis=0)
                data_frame_02.resize(np.shape(Y_val)[0], axis=0)
                data_frame_03.resize(np.shape(Y_val)[0], axis=0)
                data_frame_04.resize(np.shape(Y_val)[0], axis=0)
                data_frame_target[:] = Y_val
                data_frame_01[:] = X_01_val
                data_frame_02[:] = X_02_val
                data_frame_03[:] = X_03_val
                data_frame_04[:] = X_04_val

            elif dset_type == "test":
                data_frame_target.resize(np.shape(Y_test)[0], axis=0)
                data_frame_01.resize(np.shape(Y_test)[0], axis=0)
                data_frame_02.resize(np.shape(Y_test)[0], axis=0)
                data_frame_03.resize(np.shape(Y_test)[0], axis=0)
                data_frame_04.resize(np.shape(Y_test)[0], axis=0)
                data_frame_target[:] = Y_test
                data_frame_01[:] = X_01_test
                data_frame_02[:] = X_02_test
                data_frame_03[:] = X_03_test
                data_frame_04[:] = X_04_test    

    elapsed_time = datetime.datetime.now() - start_time  
    print("Time to build datasets: %s" % (elapsed_time))

def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        yield X1[idx], np.array([ [X2[0][i], X2[1][i], X2[2][i], X2[3][i]] for i in idx ])

def downsize_image(X, size):
    return cv2.resize(X, (128, 128), interpolation=cv2.INTER_NEAREST) 

def gen_local_batch(X1, X2, batch_size, rows, cols):
    while True:
        X1_out = split_frame(X1, batch_size, rows, cols)
        idx = np.random.choice(X1_out.shape[0], batch_size*(rows * cols), replace=False)
        F0 = split_frame(X2[:,0,:,:,:], batch_size, rows, cols)
        F1 = split_frame(X2[:,1,:,:,:], batch_size, rows, cols)
        F2 = split_frame(X2[:,2,:,:,:], batch_size, rows, cols)
        F3 = split_frame(X2[:,3,:,:,:], batch_size, rows, cols)  
        yield X1_out[idx], np.array([ [F0[i], F1[i], F2[i], F3[i]] for i in idx ])

def split_frame(X1, batch_size, rows, cols):
    img_dim = np.shape(X1)[1]
    X1_new = []
    for i in range(batch_size):
        for x in range(rows):
            for y in range(cols):
                X1_new.append(X1[i,img_dim/rows * x:img_dim/rows * (x+1), img_dim/cols * y:img_dim/cols * (y+1), :])
    
    X1_new = np.asarray(X1_new)
    

    return X1_new

def get_disc_batch(X_full_batch, X_sketch_batch, generator_model, batch_counter, patch_size,
                   image_data_format, label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_sketch_batch)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_full_batch
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    X_disc = extract_patches(X_disc, image_data_format, patch_size)

    return X_disc, y_disc


def plot_generated_batch(X_full, X_sketch, generator_model, batch_size, image_data_format, suffix):
    fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=300, facecolor='w', edgecolor='k')
    # Generate images
    X_gen = generator_model.predict(X_sketch)

    X_sketch_0 = inverse_normalization(X_sketch[:,0,:,:,:])
    X_sketch_1 = inverse_normalization(X_sketch[:,1,:,:,:])
    X_sketch_2 = inverse_normalization(X_sketch[:,2,:,:,:])
    X_sketch_3 = inverse_normalization(X_sketch[:,3,:,:,:])
    X_full = inverse_normalization(X_full)
    X_gen = inverse_normalization(X_gen)

    Xs0 = X_sketch_0[:4]
    Xs1 = X_sketch_1[:4]
    Xs2 = X_sketch_2[:4]
    Xs3 = X_sketch_3[:4]
    Xg = X_gen[:4]
    Xr = X_full[:4]

    if image_data_format == "channels_last":
        X = np.concatenate((Xs0, Xs1, Xs2, Xs3, Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] // 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=0)

    if image_data_format == "channels_first":
        X = np.concatenate((Xs, Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] // 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=1)
        Xr = Xr.transpose(1,2,0)

    if Xr.shape[-1] == 1:
        ax.imshow(Xr[:, :, 0], cmap="gray")
    else:
        ax.imshow(Xr, interpolation='nearest', aspect='auto')
    ax.axis("off")
    ax.set_aspect('auto')
    plt.savefig("../../figures/current_batch_%s.png" % suffix)
    plt.clf()
    plt.close()
