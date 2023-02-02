
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("pngdir", type=str, help='path to the folder containing resmos pngs')
parser.add_argument("propfile",type=str, help="path to the table file containing image properties")
parser.add_argument("output",type=str, help="name of the output hdf5 file" )
parser.add_argument("--overwrite", action="store_true", help="overwrite output file if it exists")
args = parser.parse_args()

import h5py
import numpy as np
import os

assert os.path.exists(args.pngdir)
assert os.path.exists(args.propfile)
assert os.path.isdir(args.pngdir)

from resonet import net

if os.path.exists(args.output) and not args.overwrite:
    print("file %s exists!, use the --overwrite flag" % args.output)

with h5py.File(args.output, "w") as h:
    imgs = net.Images(pngdir=args.pngdir, propfile=args.propfile)
    imgs.props = ["reso"]
    dset = h.create_dataset('images', dtype=np.float32, shape=(imgs.total, 1, 512, 512))
    dset_lab = h.create_dataset('labels', dtype=np.float32, shape=(imgs.total,len(imgs.props)))

    start = 0
    while start < imgs.total:
        I, L = imgs[start:start+100]
        dset[start:start+I.shape[0]] = I[:,:1]
        dset_lab[start:start+I.shape[0]] = 1/L
        start += 100
        print("Copied %d /%d"%(start, imgs.total))
