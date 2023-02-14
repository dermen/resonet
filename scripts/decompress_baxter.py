import h5py
from joblib import Parallel, delayed
import numpy as np
import sys
import glob
import os

"""
usage:
    python decompress_baxter.py baxter.4 10

Will create new files in folder baxter.4 using 10 processes
"""

DIRNAME = sys.argv[1]
NJOBS = int(sys.argv[2])


def main(jid):
    fnames = glob.glob(DIRNAME + "/compressed*.h5")

    for i_f, f in enumerate(fnames):
        if i_f % NJOBS != jid:
            continue
        fnew = f.replace("compressed", "rank")
        h = h5py.File(f, "r")
        imgs = h['images']
        labels = h['labels'][()]
        with h5py.File(fnew, "w") as hnew:
            dset = hnew.create_dataset("images", shape=imgs.shape, dtype=np.float32)
            for i_img in range(imgs.shape[0]):
                if i_img % 10==0:
                   print(f"Job{jid} Decompressing file {i_f+1}/{len(fnames)}, shot {i_img+1}/{imgs.shape[0]}") 
                dset[i_img] = imgs[i_img].astype(np.float32)
                
            hnew.create_dataset("labels", data=labels.astype(np.float32))

Parallel(n_jobs=NJOBS)(delayed(main)(jid) for jid in range(NJOBS))

