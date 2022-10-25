
import h5py
import numpy as np
import os


import net

name = "trainimages9prop.h5"


assert not os.path.exists(name)

with h5py.File(name, "w") as h:
    imgs = net.Images()
    imgs.props = ["reso", "mos", "icy1", "icy2", "B", "cell1", "cell2", "cell3", "SGnum"]
    dset = h.create_dataset('images', dtype=np.float32, shape=(imgs.total,2,512,512))
    dset_lab = h.create_dataset('labels', dtype=np.float32, shape=(imgs.total,len(imgs.props)))

    start = 0
    while start < imgs.total:
        I,L = imgs[start:start+100]
        dset[start:start+I.shape[0]] = I
        dset_lab[start:start+I.shape[0]] = L
        start += 100
        print(start)

