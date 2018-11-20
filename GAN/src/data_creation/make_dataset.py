import os
import cv2
import h5py
import parmap
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm as tqdm
import matplotlib.pylab as plt
from glob import glob

def format_image(img_path, size, nb_channels):
    """
    Load img with opencv and reshape
    """
    #print("Format: " + str(img_path))
    if nb_channels == 1:
        img = cv2.imread(img_path, 0)
        img = np.expand_dims(img, axis=-1)
    else:
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]  # GBR to RGB

    w = img.shape[1]
    img_sketch = img[:, :, :]

    img_sketch = cv2.resize(img_sketch, (size, size), interpolation=cv2.INTER_AREA)

    if nb_channels == 1:
        img_sketch = np.expand_dims(img_sketch, -1)

    img_sketch = np.expand_dims(img_sketch, 0).transpose(0, 3, 1, 2)

    return img_sketch


def build_HDF5(jpeg_dir, nb_channels, size=256):
    """
    Gather the data in a single HDF5 file.
    """

    # Put train data in HDF5
    file_name = os.path.basename(jpeg_dir.rstrip("/"))
    hdf5_file = os.path.join(data_dir, "%s_data.h5" % file_name)
    with h5py.File(hdf5_file, "w") as hfw:

        for dset_type in ["train", "test", "val"]:
            print("Build DSet " + str(dset_type))

            data_frame_target = hfw.create_dataset("%s_data_frame_target" % dset_type,
                                           (0, nb_channels, size, size),
                                           maxshape=(None, 3, size, size),
                                           dtype=np.uint8)

            data_frame_01 = hfw.create_dataset("%s_data_frame_01" % dset_type,
                                             (0, nb_channels, size, size),
                                             maxshape=(None, 3, size, size),
                                             dtype=np.uint8)

            data_frame_02 = hfw.create_dataset("%s_data_frame_02" % dset_type,
                                             (0, nb_channels, size, size),
                                             maxshape=(None, 3, size, size),
                                             dtype=np.uint8)

            data_frame_03 = hfw.create_dataset("%s_data_frame_03" % dset_type,
                                             (0, nb_channels, size, size),
                                             maxshape=(None, 3, size, size),
                                             dtype=np.uint8)

            data_frame_04 = hfw.create_dataset("%s_data_frame_04" % dset_type,
                                             (0, nb_channels, size, size),
                                             maxshape=(None, 3, size, size),
                                             dtype=np.uint8)

            list_dirs = [os.path.basename(x) for x in glob(str(jpeg_dir) + "/" + (dset_type) + "/*")]
            
            for dir_name in list_dirs:
                print("Building dir " + str(dir_name))
                list_img = [img for img in Path(jpeg_dir).glob(str(dset_type) + '/' + str(dir_name) + '/frame*.jpg')]
                list_img = [str(img) for img in list_img]
                list_img.extend(list(Path(jpeg_dir).glob('%s/*.png' % dset_type)))
                list_img = list(map(str, list_img))
                list_img = np.array(list_img)
                list_img = np.sort(list_img)
                num_files = len(list_img)
                arr_chunks = np.array(np.arange(0,num_files-8))

                for chunk_idx in tqdm(arr_chunks):
                    
                    list_img_path = list_img[np.hstack([np.array(np.arange(chunk_idx, chunk_idx+4)), chunk_idx+8])].tolist()
                    output = parmap.map(format_image, list_img_path, size, nb_channels, pm_parallel=False)
                    arr_frame_target = np.concatenate([output[4]], axis=0)
                    arr_frame_01 = np.concatenate([output[0]], axis=0)
                    arr_frame_02 = np.concatenate([output[1]], axis=0)
                    arr_frame_03 = np.concatenate([output[2]], axis=0)
                    arr_frame_04 = np.concatenate([output[3]], axis=0)

                    data_frame_target.resize(data_frame_target.shape[0] + arr_frame_target.shape[0], axis=0)
                    data_frame_01.resize(data_frame_01.shape[0] + arr_frame_01.shape[0], axis=0)
                    data_frame_02.resize(data_frame_02.shape[0] + arr_frame_02.shape[0], axis=0)
                    data_frame_03.resize(data_frame_03.shape[0] + arr_frame_03.shape[0], axis=0)
                    data_frame_04.resize(data_frame_04.shape[0] + arr_frame_04.shape[0], axis=0)

                    data_frame_target[-arr_frame_target.shape[0]:] = arr_frame_target.astype(np.uint8)
                    data_frame_01[-arr_frame_01.shape[0]:] = arr_frame_01.astype(np.uint8)
                    data_frame_02[-arr_frame_02.shape[0]:] = arr_frame_02.astype(np.uint8)
                    data_frame_03[-arr_frame_03.shape[0]:] = arr_frame_03.astype(np.uint8)
                    data_frame_04[-arr_frame_04.shape[0]:] = arr_frame_04.astype(np.uint8)

def check_HDF5(jpeg_dir, nb_channels):
    """
    Plot images with landmarks to check the processing
    """

    # Get hdf5 file
    file_name = os.path.basename(jpeg_dir.rstrip("/"))
    hdf5_file = os.path.join(data_dir, "%s_data.h5" % file_name)

    with h5py.File(hdf5_file, "r") as hf:
        data_frame_target = hf["train_data_frame_target"]
        data_frame_01 = hf["train_data_frame_01"]
        data_frame_02 = hf["train_data_frame_02"]
        data_frame_03 = hf["train_data_frame_03"]
        data_frame_04 = hf["train_data_frame_04"]
        for i in range(data_frame_target.shape[0]):
            plt.figure()
            img_target = data_frame_target[i, :, :, :].transpose(1,2,0)
            img2 = data_frame_01[i, :, :, :].transpose(1,2,0)
            img3 = data_frame_02[i, :, :, :].transpose(1,2,0)
            img4 = data_frame_03[i, :, :, :].transpose(1,2,0)
            img5= data_frame_04[i, :, :, :].transpose(1,2,0)
            img = np.concatenate((img2, img3, img4, img5, img_target), axis=1)
            if nb_channels == 1:
                plt.imshow(img[:, :, 0], cmap="gray")
            else:
                plt.imshow(img)
            plt.show()
            plt.clf()
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build dataset')
    parser.add_argument('jpeg_dir', type=str, help='path to jpeg images')
    parser.add_argument('nb_channels', type=int, help='number of image channels')
    parser.add_argument('--img_size', default=256, type=int,
                        help='Desired Width == Height')
    parser.add_argument('--do_plot', action="store_true",
                        help='Plot the images to make sure the data processing went OK')
    args = parser.parse_args()

    data_dir = "../../data"

    build_HDF5(args.jpeg_dir, args.nb_channels, size=args.img_size)

    if args.do_plot:
        check_HDF5(args.jpeg_dir, args.nb_channels)
