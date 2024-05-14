import os
import numpy as np
from tqdm import tqdm
import nrrd
import h5py


def main(file_seg):
    vol_seg , header_seg = nrrd.read(file_seg)