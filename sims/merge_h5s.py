import h5py
import sys
import glob
import os

"""
Use this method to merge the rank*.h5 files that are output by
runme_cpu.py (when using MPI mode , each rank writes a file)
"""

dirname = sys.argv[1]
fnames = glob.glob(os.path.join(dirname, "rank*h5"))
print("Combining %d files" % len(fnames))

#<KeysViewHDF5 ['images', 'labels', 'mask', 'raw_images']>
dummie_h = h5py.File(fnames[0], "r")

shapes = {}
for key in ["images", "labels", "raw_images"]:
    shapes[key] = dummie_h[key].shape[1:] 

single_dsets = ["mask", "pixel_radius_map"]

imgs_per_fname = [h5py.File(f, 'r')['labels'].shape[0] for f in fnames]
total_imgs = sum(imgs_per_fname)
print("Total number of shots=%d" % total_imgs)

Layouts = {}
for key, shape in shapes.items():
    Layouts[key] = h5py.VirtualLayout(shape=(total_imgs,) + shape, dtype=dummie_h[key].dtype)

Layouts["twoChannel"] = h5py.VirtualLayout(shape=(total_imgs, 2) + shapes['images'], dtype=dummie_h['images'].dtype)

start = 0
for i_f, f in enumerate(fnames):
    print("virtualizing file %d / %d" % (i_f+1, len(fnames)))
    nimg = imgs_per_fname[i_f]
    for key in Layouts:
        if key=="twoChannel":
            continue
        vsource = h5py.VirtualSource(f, key, shape=(nimg,) + shapes[key])
        Layouts[key][start:start+nimg] = vsource

    res_map_source = h5py.VirtualSource(f, "pixel_radius_map", shape=shapes["images"])
    im_source = h5py.VirtualSource(f, "images", shape=(nimg,) + shapes["images"])
    Layouts["twoChannel"][start:start+nimg, 0] = im_source
    for i_img in range(nimg):
        Layouts["twoChannel"][start+i_img, 1] = res_map_source
    start += nimg

master_name = os.path.join(dirname, "master.h5")
print("Saving it all to %s!" % master_name)
with h5py.File(master_name, "w") as H:
    for key in Layouts:
        H.create_virtual_dataset(key, Layouts[key])

    for key in single_dsets:
        H.create_dataset(key, data=dummie_h[key][()])

print("Done!")

