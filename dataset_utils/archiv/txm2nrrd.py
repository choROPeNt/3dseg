import os
#=====================
# Skript for converting a *.txm to a *.nrrd file
# created by Christian Düreth
# mail: christian.duereth@tu-dresden.de
# command line arguments
# --filepath path to a *.txm file
# --dtype data type for the *.nrrd file, supported uint8 uint 16 


import numpy as np
import xrmreader
#TODO install spefile, EDfFile and astropy
import nrrd
import argparse


def main(args):

    filepath = args.filepath
    dtype = args.dtype


    root = os.path.dirname(filepath)
    file = os.path.basename(filepath)
    filename, filext = os.path.splitext(file)
    
    if os.path.isfile(filepath):
        print('loading %s' %file)
        data = xrmreader.read_txrm(filepath)
        metadata = xrmreader.read_metadata(filepath)
        print('converting %s' %file)

        # change from [d,w,h] to [h,w,d]
        data = data.transpose(2,1,0)
        
        # create new header
        scale = metadata["pixel_size"]
        hnrrd  = {}
        hnrrd["space"] = 'left-posterior-superior'
        hnrrd["space directions"] = [[scale,0.,0.],[0.,scale,0.],[0.,0.,scale]]
        hnrrd["kinds"] = ['domain', 'domain', 'domain']
        hnrrd["space origin"] = [0.,0.,scale]

        # convert to datatypes
        if dtype == 'uint8':
            print('converting to %s' %dtype)
            data = np.array(data/data.max()*255,dtype=np.uint8)
            print('writting %s _uint8.nrrd' %filename)
            nrrd.write(os.path.join(root,filename + '_uint8.nrrd'),data,hnrrd)
    
    
    else:
        print('"%s" not found in root "%s"!'%(file,root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', help='path to txm file')
    parser.add_argument('--dtype', default='uint8' ,help='export datatype')
    args = parser.parse_args()

    main(args)
