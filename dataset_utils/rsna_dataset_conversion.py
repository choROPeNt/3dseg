import os
import numpy as np
import nibabel as nib
import h5py
import pydicom as dicom
import matplotlib.pylab as plt
from natsort import natsorted, ns

def main():
    seg = nib.load(os.path.join('data','rsna-2022-cervical-spine-fracture-detection','train_images','1.2.826.0.1.3680043.780.nii'))

    seg_data = np.rot90(seg.get_data())

    DCIMpath = os.path.join('data','rsna-2022-cervical-spine-fracture-detection','train_images','1.2.826.0.1.3680043.780')
    DCIMlist = natsorted(os.listdir(DCIMpath), alg=ns.IGNORECASE)
    # print(DCIMlist[::-1])
    vol_data = np.empty_like(seg_data)

    for i, file in enumerate(DCIMlist[::-1]):
        ds = dicom.dcmread(os.path.join(DCIMpath,file))
        vol_data[:,:,i] = ds.pixel_array

    with h5py.File(os.path.join('data','rsna-2022-cervical-spine-fracture-detection','train_images','1.2.826.0.1.3680043.780.hdf5'), "w") as f:

        f.create_dataset("label", data = seg_data)
        f.create_dataset("raw", data = vol_data)




if __name__ == "__main__":
    main()