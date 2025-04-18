import os
import h5py as h5
import nrrd
import argparse
import numpy as np
from skimage.measure import label
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'plasma'


def readH5(file_path, **kwargs):
    """
    Generic function to read an HDF5 file with nested groups.
    It supports up to two subgroup levels and returns the content as a nested dictionary.
    
    Parameters:
    - file_path (str): Path to the HDF5 file.
    
    Returns:
    - dict: Nested dictionary containing datasets from the file.
    """
    out = {}

    with h5.File(file_path, 'r') as f:
        for key in f.keys():
            if isinstance(f[key], h5.Group):
                out[key] = {}
                for sub_key in f[key].keys():
                    if isinstance(f[key][sub_key], h5.Group):
                        out[key][sub_key] = {}
                        for sub_sub_key in f[key][sub_key].keys():
                            out[key][sub_key][sub_sub_key] = f[key][sub_key][sub_sub_key][...]
                    else:
                        out[key][sub_key] = f[key][sub_key][...]
            else:
                out[key] = f[key][...]
    return out


def create_nrrd_header(size, spacing,space_origin, data_type='float', space='left-posterior-superior'):
    header = {
        'type': data_type,
        'dimension': len(size),
        'space dimension': len(size),
        'sizes': size,
        'space': space,
        'space origin': space_origin
    }

    if len(spacing) == len(size):
        header['space directions'] = np.diag(spacing).tolist()
        header['kinds'] = ['domain'] * len(size)

    header.update({
        'endian': 'little',
        'encoding': 'raw',
    })

    return header

def numpy_dtype_to_nrrd_dtype(dtype):
    """
    Convert NumPy dtype to NRRD dtype string.

    Parameters:
    - dtype: NumPy dtype object

    Returns:
    - NRRD dtype string
    """
    dtype_mapping = {
        np.uint8: 'uint8',
        np.uint16: 'uint16',
        np.uint32: 'uint32',
        np.uint64: 'uint64',
        np.int8: 'int8',
        np.int16: 'int16',
        np.int32: 'int32',
        np.int64: 'int64',
        np.float16: 'float16',
        np.float32: 'float32',
        np.float64: 'float64',
    }

    return dtype_mapping.get(dtype, 'unknown')


## TODO make it more generic.

def main():

    parser = argparse.ArgumentParser(description="Process a file path.")

    # Add the argument
    parser.add_argument(
        'file_path', 
        type=str, 
        help='The path to the file to be processed'
    )

     # Parse the arguments
    args = parser.parse_args()
    # Extract the directory path and file name
    directory = os.path.dirname(args.file_path)
    file_name = os.path.basename(args.file_path)
    file_name_ = file_name.split('.')

    # Print the directory and file name
    print(f'Directory path: {directory}')
    print(f'File name: {file_name}')
    print(f'File name: {file_name_}')

    file   = readH5(os.path.join(directory,file_name))

    data= {}

    for key, item in file.items():
  
        if key == "predictions":
            print(f"proccessing {key} with {item.shape} and {item.dtype}")
            
            labels = np.argmax(item,axis=0)
            
            print(labels.shape)
            
            print(f"created new array labels with {labels.shape} and {labels.dtype}")

            data_type_seg = labels.dtype
            size = labels.shape

            spacing = (1,1,1)
            space_origin = (0 , 0 , 0 )

            header_seg = create_nrrd_header(size, spacing, space_origin, data_type_seg)
           
            fileout = file_name_[0] +".pred.nrrd"
            print(f"Creating file {fileout} in {directory}")
            nrrd.write(os.path.join(directory,fileout), labels,header=header_seg)
    
        
        if key == "Volume":
            print(f"proccessing {key} with {item.shape} and {item.dtype}")
            data_type_seg = item.dtype
            size = item.shape

            spacing = (1,1,1)
            space_origin = (0 , 0 , 0 )

            header_seg = create_nrrd_header(size, spacing, space_origin, data_type_seg)
            
            fileout = file_name_[0] +".vol.nrrd"
            print(f"Creating file {fileout} in {directory}")
    
            nrrd.write(os.path.join(directory,fileout), item,header=header_seg)

if __name__ == "__main__":
    
    main()
    
    