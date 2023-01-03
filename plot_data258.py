import h5py
import matplotlib.pyplot as plt
import os




f = h5py.File(os.path.join('data','predict','Movie1_t00006_crop_gt_predictions.h5'), 'r') 
print(f.keys())
vol  = f['predictions']
# label =f['label']
print(vol)
fig, ax = plt.subplots(2)

ax[0].imshow(vol[0,5,:,:])
# ax[1].imshow(label[5])

fig.show()