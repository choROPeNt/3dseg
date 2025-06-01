import torch
import h5py
import os
import numpy as np
from tqdm import tqdm



from torch3dseg.utils.data import readH5
from torch3dseg.descriptors.descriptors import compute_s2
from torch3dseg.utils.utils import expand_as_one_hot

from time import time


from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline




def main():
    root = "/data/horse/ws/dchristi-3dseg/predict/BIIAX/model_2025-05-19_00/best_checkpoint/"
    file_list = ["285_05-layer_01.pred.labels.h5","285_05-layer_02.pred.labels.h5","285_05-layer_03.pred.labels.h5",
                 "285_10-layer_01.pred.labels.h5","285_10-layer_02.pred.labels.h5","285_10-layer_03.pred.labels.h5",
                 "285_37-layer_01.pred.labels.h5","285_37-layer_02.pred.labels.h5","285_37-layer_03.pred.labels.h5"]

    for file in file_list:
        # --- Step 1: Load data lazily and clear ASAP ---
        data = readH5(os.path.join(root, file))
        file_name = file.split('.')[0]
        print(file_name)

        # Move to int64 only if needed, avoid int64 if possible
        ms = torch.tensor(data["predictions_labels"][()], dtype=torch.int64).unsqueeze(0)
        del data  # Free early
        torch.cuda.empty_cache()

        ms = expand_as_one_hot(ms, C=3)[:, 1:, ...]  # Shape: [1, 2, D, H, W]
        D, H, W = ms.shape[2:]
        print(f"Volume size: {D}x{H}x{W}")
        print(f"Dataype of ms: {ms.dtype}")

        # Descriptor setup
        space = 75
        descriptor = compute_s2(normalize=True, limit_to=space).to("cuda")

        # Patch and stride setup
        target_patch = 150
        patch_D = min(D, target_patch)
        patch_H = patch_W = target_patch
        stride_D = patch_D // 2 if D >= patch_D else patch_D
        stride_H = stride_W = patch_H // 2
        print(f"Patches: {patch_D}x{patch_H}x{patch_W}, Strides: {stride_D},{stride_H},{stride_W}")

        # --- Step 2: Patch-wise running mean ---
        running_sum = None
        count = 0
        print(f"Analyzing {file_name}...")

        with torch.no_grad():
            for z in range(0, D - patch_D + 1, stride_D):
                for y in range(0, H - patch_H + 1, stride_H):
                    for x in range(0, W - patch_W + 1, stride_W):
                        patch = ms[:, :, z:z+patch_D, y:y+patch_H, x:x+patch_W].to("cuda")
                        s2_patch = descriptor(patch).squeeze(1).cpu()  # shape: [1, Z, Y, X] -> [Z, Y, X]

                        if running_sum is None:
                            running_sum = torch.zeros_like(s2_patch)

                        running_sum += s2_patch
                        count += 1

                        del patch, s2_patch
                        torch.cuda.empty_cache()

        s2_mean = (running_sum / count).numpy()
        del running_sum, ms
        torch.cuda.empty_cache()

        with h5py.File(os.path.join(root,file_name + ".descriptor-S2.h5"),"w") as f:
            f.create_dataset("S2_mean",data=s2_mean,compression="gzip")

if __name__ == "__main__":
    main()