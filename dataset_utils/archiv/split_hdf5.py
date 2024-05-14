import os 
import numpy as np
import h5py



def extract_blocks_2d(a, blocks, keep_as_view=False):
    w,h = a.shape
    m, n = blocks
    
    b1,b2 =w//m , h//n


    if w % b1 == 0:
        a = a
    else:
        a = a[:-1,:]
    
    if h % b2 == 0:
        a = a
    else:
        a = a[:,:-1]


    if keep_as_view==0:
        return a.reshape(w//b1,b1,h//b2,b2).swapaxes(1,2).reshape(-1,b1,b2)
    else:
        return a.reshape(w//b1,b1,h//b2,b2).swapaxes(1,2)

def extract_blocks_3d(a, blocks, keep_as_view=False):
    
    d, w, h = a.shape
    k, m, n = blocks
    
    b0, b1, b2 = d//k, w//m , h//n
    
    if d % b0 == 0:
        a = a
    else:
        a = a[:-1,:,:]

    if w % b1 == 0:
        a = a
    else:
        a = a[:,:-1,:]
    
    if h % b2 == 0:
        a = a
    else:
        a = a[:,:,:-1]

    if keep_as_view==0:
        return a.reshape(d//b0,b0,w//b1,b1,h//b2,b2).transpose(0,2,4,1,3,5).reshape(-1,b0,b1,b2)
    else:
        return a.reshape(d//b0,b0,w//b1,b1,h//b2,b2).transpose(0,2,4,1,3,5)

def extract_blocks_4d(a, blocks, keep_as_view=False):
    c,d,w,h = a.shape
    k, m, n = blocks
    
    b0, b1, b2 = d//k, w//m , h//n
    
    if d % b0 == 0:
        a = a
    else:
        a = a[:,:-1,:,:]

    if w % b1 == 0:
        a = a
    else:
        a = a[:,:,:-1,:]
    
    if h % b2 == 0:
        a = a
    else:
        a = a[:,:,:,:-1]


    if keep_as_view==0:
        return a.reshape(c,d//b0,b0,w//b1,b1,h//b2,b2).transpose(1,3,5,0,2,4,6).reshape(-1,c,b0,b1,b2)
    else:
        return a.reshape(c,d//b0,b0,w//b1,b1,h//b2,b2).transpose(1,3,5,0,2,4,6).swapaxes(3,4)


def main(filepath,blocks):
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            data[key] = np.array(f[key])

    out = {}
    for key in data.keys():
        print(key)
        array = data[key]
        print(array.shape)
        ndims = len(array.shape)

        if ndims == 3:
            array = extract_blocks_3d(array, blocks, keep_as_view=False)
        elif ndims == 4:
            array = extract_blocks_4d(array, blocks, keep_as_view=False)
        print(array.shape)
        out[key] = array
        


    file_name   = os.path.splitext(os.path.basename(filepath))[0]
    root        = os.path.dirname(filepath)


    print(file_name,root)
    
    for n in range(np.prod(blocks)):
        with h5py.File(os.path.join(root,(file_name +'-'+ str(n).zfill(2) + '.hdf5')), 'w') as fout:
            for key in out.keys():
                
                array = out[key]
                ndims = len(array.shape)
                print(key,ndims)
                if (ndims-1) == 3:
                    fout.create_dataset(key, data = array[n,:,:,:])
                elif (ndims-1) == 4:
                    fout.create_dataset(key, data = array[n,:,:,:,:])

if __name__ == "__main__":
    filepath = './data/Neapel/neapel_001.hdf5'
    blocks = (1,2,2)
    main(filepath,blocks)