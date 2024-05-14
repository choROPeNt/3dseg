import h5py
import numpy as np
import os
import argparse

def main(file,file_out):


    with h5py.File(file, 'r') as f:
        raw = f['raw']
        label = f['label']
        obj_ids = np.unique(label)
        obj_ids = obj_ids[1:]
        masks = label == obj_ids[:, None, None, None]
    
        out = {}
        out['raw'] = raw
        out['label'] = label
        out['masks'] = masks

        with h5py.File(file_out, 'w') as fout:
            for key in out.keys():
                fout.create_dataset(key, data = out[key])
        
    


if __name__ == "__main__":
    #TODO solve os error file not found. when passsing os.path object to h5py.File
    print(os.getcwd())
    # file = os.path.join('.','data','BIIAX','val','Biax_type285_img_0001-0050.hdf5')
    file = os.path.join('.','Biax_type285_img_1751-1800.hdf5')

    file_out = os.path.join('.','Biax_type285_img_1751-1800_channel4.hdf5')
    print(os.path.exists(file))
    print(os.path.isfile(file))
    # file = r'Biax_type285_img_1751-1800.hdf5'
    print(file)
    main(file,file_out)
