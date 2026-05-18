import h5py as h5
import nrrd
import argparse
import numpy as np
from pathlib import Path
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


def main():

    parser = argparse.ArgumentParser(description="Extract a dataset from an HDF5 file and save it as NRRD.")

    parser.add_argument('file_path', type=str, help='Path to the HDF5 file')
    parser.add_argument('--key', '-k', type=str, required=True,
                        help='HDF5 dataset key to extract (use "/" for nested groups, e.g. "group/dataset")')
    parser.add_argument('--argmax', action='store_true',
                        help='Apply argmax along axis 0 (use for one-hot / channel-first prediction arrays)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output NRRD file path (default: same directory as input, named <stem>.<key>.nrrd)')

    args = parser.parse_args()

    in_path = Path(args.file_path).resolve()
    stem = in_path.name.split('.')[0]

    print(f'File: {in_path}')
    print(f'Key:  {args.key}')

    with h5.File(in_path, 'r') as f:
        if args.key not in f:
            raise KeyError(f"Key '{args.key}' not found in file. Available keys: {list(f.keys())}")
        node = f[args.key]
        if not isinstance(node, h5.Dataset):
            raise ValueError(f"'{args.key}' is a group, not a dataset. Specify a leaf dataset key.")
        data: np.ndarray = node[...]

    print(f'Loaded array with shape={data.shape}, dtype={data.dtype}')

    if args.argmax:
        data = np.argmax(data, axis=0)
        print(f'After argmax: shape={data.shape}, dtype={data.dtype}')

    spacing = (1, 1, 1)
    space_origin = (0, 0, 0)
    header = create_nrrd_header(data.shape, spacing, space_origin, numpy_dtype_to_nrrd_dtype(data.dtype.type))

    if args.output:
        out_path = Path(args.output)
    else:
        safe_key = args.key.replace('/', '_')
        out_path = in_path.parent / f'{stem}.{safe_key}.nrrd'

    print(f'Writing {out_path}')
    nrrd.write(str(out_path), data, header=header)

if __name__ == "__main__":
    
    main()
    
    