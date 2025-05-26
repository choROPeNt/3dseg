import torch
import h5py
import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['image.cmap'] = 'plasma'


from torch3dseg.utils.data import readH5
from torch3dseg.descriptors.descriptors import compute_s2
from torch3dseg.utils.utils import expand_as_one_hot

from time import time
import nrrd

from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline




def main():
    root = "/data/horse/ws/dchristi-3dseg/predict/BIIAX/model_2025-05-19_00/best_checkpoint/"
    file_list = ["285_05-layer_01.pred.labels.h5","285_05-layer_01.pred.labels.h5","285_05-layer_01.pred.labels.h5",
                 "285_10-layer_01.pred.labels.h5","285_10-layer_01.pred.labels.h5","285_10-layer_01.pred.labels.h5",
                 "285_37-layer_01.pred.labels.h5","285_37-layer_01.pred.labels.h5","285_37-layer_01.pred.labels.h5"]

    for file in file_list:
        data = readH5(os.path.join(root,file))

        file_name = file.split('.')[0]
        print(file_name)    
        ms = torch.tensor(data["predictions_labels"][()],dtype=torch.int64).unsqueeze(0)
        ms = expand_as_one_hot(ms,C=3)
        ms = ms[:, 1:, ...] # just select the weft and fill channel 

        del data

        device = torch.device("mps")
        space = 75

        descriptor = compute_s2(normalize=True,limit_to=space).to(device)

        ms = ms.to(device)
        _, C, D, H, W = ms.shape


        # Target patch sizes
        target_patch = 150
        patch_D = min(D, target_patch)
        patch_H = target_patch
        patch_W = target_patch
        print(patch_D,patch_H,patch_W)
        # Strides
        stride_D = patch_D // 2 if D >= patch_D else patch_D
        stride_H = patch_H // 2
        stride_W = patch_W // 2
        print(stride_D,stride_H,stride_W)

        s2_list = []
        print(f"Analysing {file_name}")
        with torch.no_grad():
            for z in range(0, D - patch_D + 1, stride_D):
                for y in range(0, H - patch_H + 1, stride_H):
                    for x in range(0, W - patch_W + 1, stride_W):
                        patch = ms[:, :, z:z+patch_D, y:y+patch_H, x:x+patch_W]
                        s2_patch = descriptor(patch)
                        s2_list.append(s2_patch.cpu())

        # Stack and compute mean
        s2_mean = torch.stack(s2_list).squeeze(1).mean(dim=0).cpu().numpy()

        with h5py.File(os.path.join(root,file_name + ".descriptor-S2.h5"),"w") as f:
            f.create_dataset("S2_mean",data=s2_mean,compression="gzip")