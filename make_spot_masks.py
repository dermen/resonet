#!/usr/bin/env libtbx.python

from mpi4py import MPI
COMM = MPI.COMM_WORLD

import numpy as np
from PIL import Image
import glob
import os


from simemc import utils

fnames = glob.glob("../*A.png")
maskdir="/global/cfs/cdirs/m3992/png/masks"

params = utils.stills_process_params_from_file("spots.phil")
trust_mask = np.load("trust_mask.npy")

def load_image(f):
    im = Image.open(f)
    im = np.reshape(im.getdata(), (546,518))
    return im


for i,f in enumerate(fnames):
    if i% COMM.size != COMM.rank:
        continue
    img = load_image(f)
    spot_mask = utils.dials_find_spots(\
        img, params, trusted_flags=trust_mask)
    mask_name =maskdir + "/%s" % os.path.basename(f).replace(".png", ".npy")
    if COMM.rank==0:
        print(i, mask_name)
    np.save(mask_name, spot_mask)

