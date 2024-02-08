# from mpi4py import MPI
# import h5py as h5
# import os
# import numpy as np

# rank = MPI.COMM_WORLD.rank

# print(rank)
# pred_file = "/Volumes/LaCie/scratch/160_10-layer/volumes/00/160_10-layer_00_rot.h5"

# pred = {}
# with h5.File(os.path.join(pred_file),"r",driver='mpio', comm=MPI.COMM_WORLD) as f:
#     # print(f.keys())
#     for key in f.keys():
#         # print(key)
#         if key == "raw":
#             pred[key] = f[key][:]
#             # print(f[key].shape)
#             # print(f[key].nbytes/1024**2)
#             # print(f[key].dtype)

# print(pred["raw"].shape)
# print(pred["raw"].nbytes)
# print(pred["raw"].dtype)

from mpi4py import MPI
import h5py
import numpy as np



rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)

f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)

dset = f.create_dataset('test', (4,1000, 1000, 1000), dtype='i',
                        # chunks=(1,1000, 1000, 1000), 
                        compression="gzip")
dset[rank] = np.full((1000, 1000, 1000), rank)

f.close()