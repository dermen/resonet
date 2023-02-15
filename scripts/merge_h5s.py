import h5py
import glob
import os

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("dirnames", nargs="+", type=str)
parser.add_argument("outname", type=str)
args = parser.parse_args()

"""
Use this method to merge the rank*.h5 files that are output by
runme_cpu.py (when using MPI mode , each rank writes a file)
"""

fnames = []
for dirname in args.dirnames:
    fnames += glob.glob(os.path.join(dirname, "rank*h5"))

print("Combining %d files" % len(fnames))

dummie_h = h5py.File(fnames[0], "r")

shapes = {}
for key in ["images_mean", "images", "labels"]:
    try:
        shapes[key] = dummie_h[key].shape[1:] 
    except KeyError:
        pass


imgs_per_fname = [h5py.File(f, 'r')['labels'].shape[0] for f in fnames]
total_imgs = sum(imgs_per_fname)

Layouts = {}
for key, shape in shapes.items():
    Layouts[key] = h5py.VirtualLayout(shape=(total_imgs,) + shape, dtype=dummie_h[key].dtype)

start = 0
for i_f, f in enumerate(fnames):
    print("virtualizing file %d / %d" % (i_f+1, len(fnames)))
    nimg = imgs_per_fname[i_f]
    for key in Layouts:
        vsource = h5py.VirtualSource(f, key, shape=(nimg,) + shapes[key])
        Layouts[key][start:start+nimg] = vsource

    #im_source = h5py.VirtualSource(f, "images", shape=(nimg,) + shapes["images"])
    start += nimg

print("Saving it all to %s!" % args.outname) #master_name)
print("Total number of shots=%d" % total_imgs)
with h5py.File(args.outname, "w") as H:
    for key in Layouts:
        H.create_virtual_dataset(key, Layouts[key])

print("Done!")

