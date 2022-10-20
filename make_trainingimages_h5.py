
import net
import h5py
import numpy as np

name = "trainimages.h5"

import os
assert not os.path.exists(name)

with  h5py.File("trainimages.h5", "w") as h:
    dset = h.create_dataset('images', dtype=np.float32, shape=(imgs.total,2,512,512))
    dset_lab = h.create_dataset('labels', dtype=np.float32, shape=(imgs.total,1))

    start = 0
    imgs = net.Images()
    while start < imgs.total:
        I,L = imgs[start:start+100]
        dset[start:start+I.shape[0]] = I
        dset_lab[start:start+I.shape[0]] = L
        start += 100
        print(start)
