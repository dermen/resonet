
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("pngdir", type=str, help='path to the folder containing resmos pngs')
parser.add_argument("propfile",type=str, help="path to the table file containing image properties")
parser.add_argument("output",type=str, help="name of the output hdf5 file" )
parser.add_argument("--overwrite", action="store_true", help="overwrite output file if it exists")
args = parser.parse_args()

import h5py
import time
import numpy as np
import os

assert os.path.exists(args.pngdir)
assert os.path.exists(args.propfile)

if os.path.exists(args.output) and not args.overwrite:
    print("file %s exists!, use the --overwrite flag" % args.output)

from loaders import PngDset
png_dset = PngDset(pngdir=args.pngdir, propfile=args.propfile, convert_res=False, invert_res=False,
                   dev="cpu", reso_only_mode=False)
Nimg = len(png_dset)

label_names = list(png_dset.labels)
Nlab = len(label_names)

with h5py.File(args.output, "w") as h:
    imgs_dset = h.create_dataset('images', dtype=np.float32, shape=(Nimg, 512, 512))

    labels_dset = h.create_dataset('labels', dtype=np.float32, shape=(Nimg, Nlab))
    labels_dset.attrs["names"] = label_names
    np.savez(args.output + ".npz", pdbid_map=png_dset.pdbid_map, stolid_map=png_dset.stolid_map)
    #labels_dset.attrs["pdbid_map"] = png_dset.pdbid_map
    #labels_dset.attrs["stolid_map"] = png_dset.stolid_map

    geom = [200, 0.075, 0.977794, 4150, 4371]
    geom_dset = h.create_dataset("geom", data=[geom]*Nimg, dtype=np.float32)
    geom_dset.attrs["names"] = ["detdist", "pixsize", "wavelen", "xdim", "ydim"]

    ttot = 0
    for i_img in range(Nimg):
        if i_img > 0 and i_img % 10 == 0:
            tper = ttot / (i_img+1)
            tremain = (Nimg-i_img+1) * tper
            print("Done with shot %d/%d. Est. time remaining=%.2f sec." % (i_img+1, Nimg, tremain))
        # get image data and resolution
        t = time.time()
        img_dat, img_lab = map(lambda x: x.numpy(), png_dset[i_img])
        # save to image hdf5 file
        imgs_dset[i_img] = img_dat
        labels_dset[i_img] = img_lab[0]
        ttot += (time.time()-t)

    print("Done.")
