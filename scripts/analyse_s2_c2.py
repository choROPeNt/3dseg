import os
import torch
from torch.nn import functional as F

import numpy as np

from torch3dseg.utils.data import readH5
from torch3dseg.descriptors.descriptors import compute_s2
from torch3dseg.utils.utils import expand_as_one_hot
from scipy.ndimage import label

import h5py



def main():

    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        device = torch.device("cuda")
        torch.cuda.empty_cache()  # Clear CUDA cache to free up memory
    elif torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon GPU.")
        device = torch.device("mps")
    else:
        print("No GPU available. Using CPU.")
        device = torch.device("cpu")

    file_list = [
                "/data/horse/ws/dchristi-3dseg/data/NEAPEL/#10.2_gelege.h5",
                "/data/horse/ws/dchristi-3dseg/data/NEAPEL/#11.3_geflecht.h5"
                 ]

    descrptor_s2 = compute_s2(limit_to=128).to(device)
    down_sample = False

    for file in file_list:
        root = os.path.dirname(file)
        filename = os.path.splitext(os.path.basename(file))[0]  # filename without .h5

        print(f"Root: {root}")
        print(f"Filename (no ext): {filename}")


        if file.endswith(".h5"):
            data = readH5(file)


            for key, item in data.items():
                print(f"Key: {key}, Shape: {item.shape}, Type: {type(item)}")
                if key == "material_id":
                    ms_np = data[key]

       
        # Convert ms to a 5D tensor with shape [1, 1, Z, Y, X]
        ms = torch.tensor(ms_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # Print original tensor shape and memory usage
        original_mem_mb = ms.element_size() * ms.nelement() / (1024 ** 2)
        print(f"Input shape: {ms.shape}, Device: {ms.device}, Memory: {original_mem_mb:.2f} MB")

        # Apply 3D max pooling with kernel size and stride of 2 (halves spatial dimensions)
        if down_sample:
            ms = F.max_pool3d(ms, kernel_size=2, stride=2)

            # Print downsampled shape and new memory usage
            downsampled_mem_mb = ms.element_size() * ms.nelement() / (1024 ** 2)
            print(f"Downsampled shape: {ms.shape}, Memory: {downsampled_mem_mb:.2f} MB")

        # Compute the 2-point correlation descriptor
        s2 = descrptor_s2(ms)

        # Print shape of resulting descriptor
        print(f"S2 shape: {s2.shape}")
        # Convert to numpy for further processing    

        s2_np = s2.squeeze(0).cpu().numpy()


        ######################## cluster

         # Label input (assumed binary volume)
        ms_inst_np, num_labels = label(ms_np)  # shape: [D, H, W]
        ms_inst = torch.tensor(ms_inst_np, dtype=torch.int64, device=device).unsqueeze(0)  # [1, D, H, W]

        print(f"Number of instances: {num_labels}")

        # Init accumulator
        c2_sum = None
        valid_count = 0

        # Loop over instance labels (exclude background label 0)
        for i in range(1, num_labels + 1):
            mask = (ms_inst == i).float()  # [1, D, H, W], float for FFT

            # Skip small or empty masks
            if mask.sum() < 5:
                continue

            mask = mask.unsqueeze(0)  # -> [1, 1, D, H, W]
            s2_i = descrptor_s2(mask)

            if c2_sum is None:
                c2_sum = s2_i.clone()
            else:
                c2_sum += s2_i

            valid_count += 1

        # Final normalized cluster correlation
        c2 = c2_sum / valid_count if valid_count > 0 else torch.zeros_like(c2_sum)
        c2 = torch.clamp(c2, min=0.0)
        print(f"C2 shape: {c2.shape}, computed from {valid_count} instances")


        out =  {"s2": s2.squeeze(0).cpu().numpy(),"c2": c2.squeeze(0).cpu().numpy()}



        with h5py.File(os.path.join(root,filename + ".descriptor.h5"),"w") as f:
            for key, item in out.items():
                print(f"Writing {key} with shape {item.shape} to HDF5")
                f.create_dataset(key, data=item, compression="gzip")
      


if __name__ == "__main__":
   main()

 








