import numpy as np
import h5py

def generate_split_hd5(dset, load=["train", "val", "test"], size = 512, nb_channels = 3):    
    
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
