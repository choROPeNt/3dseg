import os
import h5py
import nrrd
import argparse
import numpy as np
import time as time
from tqdm import tqdm

import matplotlib.pyplot as plt

def main(file_pred,file_seg):
    _ , header_seg = nrrd.read(file_seg)
    pred_file = h5py.File(file_pred, 'r')
    c,d,h,w = pred_file['predictions'].shape

    print(pred_file['predictions'].dtype,pred_file['predictions'].shape)

    out_arr = np.empty((w,h,d),dtype=np.ushort)
    print('output file memory consumption: %.3f  Gbyte' %(out_arr.nbytes/1024**3))
    ## 

    ##TODO implement np.argmax(array,dim=0) !!!!!!
    for slice in tqdm(range(d)):
        ## load patch
        pred_arr = np.array(pred_file['predictions'][:,slice,:,:])
        
        pred_arr_out = np.empty((h,w))

        for cls in range(c):
            pred_arr[cls,:,:] = np.where(pred_arr[cls,:,:] >= .5, cls, 0)
            pred_arr_out = np.where(pred_arr[cls,:,:] >= .5, cls, pred_arr_out).astype(np.ushort)

        out_arr[:,:,slice] = pred_arr_out.transpose()

        # fig, axs = plt.subplots(2,2)
        # axs[0][0].imshow(pred_arr_out)
        # axs[0][1].imshow(pred_arr[1,:,:])
        # axs[1][0].imshow(pred_arr[2,:,:])
        # axs[1][1].imshow(pred_arr[3,:,:])

        plt.show()

    ##TODO change this here no hard coding
    header_seg['Segment0_Extent'] = '0 990 0 1012 0 993'
    header_seg['Segment1_Extent'] = '0 990 0 1012 0 993'
    header_seg['Segment2_Extent'] = '0 990 0 1012 0 993'
    nrrd.write('output.nrrd', out_arr, header_seg)



if __name__ == "__main__":
    file_pred   = './predict/NEAPEL/neapel_001_all_predictions.h5'
    file_seg    = './data/NEAPEL/neapel_001/Segmentation17_05_23.nrrd'
    
    start_time = time.time()
    main(file_pred,file_seg)
    print("--- %s seconds ---" % (time.time() - start_time))
    