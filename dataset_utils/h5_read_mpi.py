import numpy as np
import os
import h5py



def main(file,key="predictions",**kwargs):
    
    assert os.path.exists(file)


    with h5py.File(file,"r+") as f:
        ## load data
        data = np.argmax(f[key][:],axis=0)
        if f["labels"]:
            print("deleting old labels")
            del f["labels"]
   
        f.create_dataset("labels",data=data.astype(np.uint8))


    





if __name__=="__main__":

    file = "/Volumes/data/BIIAX_model_00-lowres-2024-02-07/160_10-layer_00_rot_0000-0230_0230-0460_predictions-test.h5"
    main(file)