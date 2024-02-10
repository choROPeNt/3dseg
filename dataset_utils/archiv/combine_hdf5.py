import os
import numpy as np
import h5py
import argparse


def main(dir):
    # load first file to check

    # vol = np.empty()
    # seg = np.empty()
    for n, file in enumerate(os.listdir(dir)):

        print(file)
        f = h5py.File(os.path.join(dir,file), 'r')
        keys = list(f.keys())
        ## create out data
        if n == 0:
            out = {}
            for key in keys:
                w,h = f[key].shape
                slices = len(os.listdir(dir))
                out[key] = np.empty([slices,w,h])

        for key in keys:
            out[key][n,:,:]=f[key]
    file_out = os.path.split(dir)[-1]+'.hdf5'
    print(file_out)
    dir_out = os.path.split(dir)[:-1]
    print(dir_out[0])
    with h5py.File(os.path.join(dir_out[0],file_out), "w") as f:
        for key in keys:
            f.create_dataset(key, data = out[key])

if __name__=="__main__":

    DIR = 'C:\\Users\\dchristi\\Documents\\Projekte\\MachineLearning\\3dseg\\data\\BIIAX\\Biax_type285_img_1751-1800'
    main(dir=DIR)

