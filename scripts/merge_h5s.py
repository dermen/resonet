import h5py
import glob
import os
import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("dirnames", nargs="+", type=str, help="output folders from runme.py or runme_joblib.py")
parser.add_argument("outname", type=str, help="name of the  master file")
parser.add_argument("--shuffle", action="store_true", help="optionally shuffle the input files before virtualizing (influences order in master file)")
parser.add_argument("--moreKeys", nargs="+", type=str, default=[], help="names of additional datasets to virtualize. These should be present in all rank* files!")
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

#has_geom = "geom" in list(dummie_h.keys())
#ngeom_params = 0
#if has_geom:
#    ngeom_params = dummie_h["geom"].shape[1]

shapes = {}
for key in ["images_mean", "images", "labels", "full_maximg", "geom"] + args.moreKeys:
    try:
        shapes[key] = dummie_h[key].shape[1:] 
    except KeyError:
        pass

imgs_per_fname = [h5py.File(f, 'r')['labels'].shape[0] for f in fnames]
total_imgs = sum(imgs_per_fname)

Layouts = {}
for key, shape in shapes.items():
    Layouts[key] = h5py.VirtualLayout(shape=(total_imgs,) + shape, dtype=dummie_h[key].dtype)

Sources = {}
records = []
start = 0
for i_f, f in enumerate(fnames):
    print("virtualizing file %d / %d" % (i_f+1, len(fnames)))
    nimg = imgs_per_fname[i_f]
    for key in Layouts:
        #if key == "geom":
        #    continue
        vsource = h5py.VirtualSource(f, key, shape=(nimg,) + shapes[key])
        Layouts[key][start:start+nimg] = vsource
        Sources[(key, f)] = vsource

    #if ngeom_params:
    #    geom_source = h5py.VirtualSource(f, "geom", shape=(ngeom_params,))
    #    for i_img in range(nimg):
    #        Layouts["geom"][start+i_img] = geom_source

    #    Sources[("geom", f)] = geom_source

    start += nimg

print("Saving it all to %s!" % args.outname) #master_name)
print("Total number of shots=%d" % total_imgs)
with h5py.File(args.outname, "w") as H:
    for key in Layouts:
        vd = H.create_virtual_dataset(key, Layouts[key])
        for attr in ["names", "pdbmap"]:
            if attr in dummie_h[key].attrs:
                vd.attrs[attr] = dummie_h[key].attrs[attr]

        #if key == "labels":
        #    vd.attrs["names"] = label_names

#if args.shuffle:
#    Sources = {}
#    if ngeom_params:
#        Sources["geom"] =h5py.VirtualSource(args.outname, "geom", shape=(total_imgs,ngeom_params))
#    for key in shapes:
#        Sources[key] = h5py.VirtualSource(args.outname, key, shape=(total_imgs,) + shapes[key])
#
#    names = set(Sources).intersection(Layouts)
#    print("Shuffle virtual sets:\n\t", names)
#    order = np.random.permutation(total_imgs)
#    for i_new, i_old in enumerate(order):
#        for name in names:
#            Layouts[name][i_new] = Sources[name][i_old]
#
#    with h5py.File(args.outname, "r+") as H:
#        for key in Layouts:
#            H.create_virtual_dataset(key+".shuff", Layouts[key])

print("Done!")
