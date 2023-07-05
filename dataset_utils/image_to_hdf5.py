import os 
from PIL import Image
import argparse
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def main(imgs_folder,dir_out,file_out,masks_folder=None):

    image_list = os.listdir(imgs_folder)

    w , h = np.array(Image.open(os.path.join(imgs_folder,image_list[0]))).shape

    d  = len(image_list)

    raw = np.empty([d,w,h])
    print('created empty array "raw" of size:', raw.shape)
    out = {}

    for n in tqdm(range(len(image_list))):
        
        raw[n,:,:] = np.array(Image.open(os.path.join(imgs_folder,image_list[n])))

    
    out['raw'] = raw

    if masks_folder:
        masks_list = os.listdir(masks_folder)  
        w , h = np.array(Image.open(os.path.join(masks_folder,masks_list[0]))).shape

        d  = len(masks_list)

        masks = np.empty([d,w,h])
        print('created empty array "masks" of size:', masks.shape)
        print(masks.shape == raw.shape)
        assert (masks.shape == raw.shape)

        for n in tqdm(range(len(masks_list))):
        
            masks[n,:,:] = np.array(Image.open(os.path.join(masks_folder,masks_list[n])))()


        print('checking for labels')
        obj_ids = np.unique(masks)
        print(obj_ids)
        obj_ids = obj_ids[:]
        print('found %s labels' %len(obj_ids))
        

        masks_bool = masks == obj_ids[:, None, None, None]      

        out['masks'] = masks_bool


    print('saving hdf5-file: %s' %(file_out + '.hdf5'))
    with h5py.File(os.path.join(dir_out,(file_out + '.hdf5')), "w") as f:
        for key in out.keys():
            f.create_dataset(key, data = out[key])


if __name__ == "__main__":
    imgs_folder = './data/BIIAX/img_0916-1831'
    # masks_folder = './data/pore-detection/raw/mask'
    masks_folder = ''
    dir_out = './data/BIIAX/'
    file_out = 'Biay_type285_img_0916-1831_raw'

    if os.path.exists(imgs_folder) and os.path.exists(masks_folder):
        main(imgs_folder,dir_out,file_out,masks_folder=masks_folder)
    elif os.path.exists(imgs_folder):
        main(imgs_folder,dir_out,file_out)
    else:
        print('%s does not exist' %imgs_folder)