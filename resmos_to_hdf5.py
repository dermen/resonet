
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
                   dev="cpu")
Nimg = len(png_dset)
labels = []
imgs = []

with h5py.File(args.output, "w") as h:
    imgs_dset = h.create_dataset('images', dtype=np.float32, shape=(Nimg, 1, 512, 512))
    all_img_res = np.zeros((Nimg,1), np.float32)
    all_img_rad = np.zeros((Nimg,1), np.float32)
    tpershot = []
    for i_img in range(Nimg):
        if i_img > 0 and i_img % 10 == 0:
            tper = np.mean(tpershot)
            tremain = (Nimg-i_img+1) * tper
            print("Done with shot %d/%d. Est. time remaining=%.2f sec." % (i_img+1, Nimg, tremain))
        # get image data and resolution
        t = time.time()
        img_dat, img_res = map(lambda x: x.numpy(), png_dset[i_img])
        # save to image hdf5 file
        imgs_dset[i_img] = img_dat
        all_img_res[i_img] = img_res
        tpershot.append( time.time()-t)

        # convert to radius
        all_img_rad[i_img, 0] = png_dset._convert_res2rad(img_res[0])

    h.create_dataset("rad", data=all_img_rad, dtype=np.float32)
    h.create_dataset("res", data=all_img_res, dtype=np.float32)
    h.create_dataset("one_over_rad", data=1/all_img_rad, dtype=np.float32)
    h.create_dataset("one_over_res", data=1/all_img_res, dtype=np.float32)
    geom_data = np.zeros((Nimg, 5), dtype=np.float32)

    print("Storing geom data as a dataset")
    detdist = 200  # mm
    pixsize = 0.075  # mm
    wavelen = 0.977794  # angstrom
    xdim, ydim = 4150, 4371
    geom_data[:] = np.array([detdist, pixsize, wavelen, xdim, ydim])
    h.create_dataset("geom", data=geom_data, dtype=np.float32)
    print("Done.")
