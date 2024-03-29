import h5py
from joblib import Parallel, delayed
from argparse import ArgumentParser
import glob
import os
import numpy as np


"""
usage:
    python decompress_baxter.py baxter.4 10

Will create new files in folder baxter.4 using 10 processes
"""


def proc_main(jid, args):

    fnames = glob.glob(args.dirname + "/compressed*.h5")

    for i_f, f in enumerate(fnames):
        if i_f % args.njobs != jid:
            continue
        fnew = f.replace("compressed", "rank")
        try:
            h = h5py.File(f, "r")
        except:
            continue
        imgs = h['images']
        labels = h['labels'][()]
        try:
            geom = h['geom'][()]
        except KeyError:
            geom = None
        with h5py.File(fnew, "w") as hnew:
            dset = hnew.create_dataset("images", shape=imgs.shape, dtype=np.float32)
            for i_img in range(imgs.shape[0]):
                if i_img % 10==0:
                   print(f"Job{jid} Decompressing file {i_f+1}/{len(fnames)}, shot {i_img+1}/{imgs.shape[0]}")
                dset[i_img] = imgs[i_img].astype(np.float32)

            lab_dset = hnew.create_dataset("labels", data=labels.astype(np.float32))
            if geom is not None:
                geom_dset = hnew.create_dataset("geom", data=geom.astype(np.float32))
                geom_dset.attrs['names'] = h['geom'].attrs['names']

            if args.names is None:
                lab_dset.attrs['names'] = h['labels'].attrs['names']
                lab_dset.attrs['pdbmap'] = h['labels'].attrs['pdbmap']
            else:
                lab_dset.attrs["names"] = args.names


def main():
    parser = ArgumentParser()
    parser.add_argument("dirname", type=str)
    parser.add_argument("--njobs", type=int, default=4)
    parser.add_argument("--names", type=str, default=None, nargs='+')
    parser.add_argument("--ranks", action="store_true", help="use this option to decompress files named rank*.h5")
    args = parser.parse_args()
    if args.ranks:
        fnames = glob.glob(args.dirname + "/rank*.h5")
        for i, f in enumerate(fnames):
            print("Copyfile", i + 1, "/", len(fnames))
            f2 = f.replace("rank", "compressed")
            os.rename(f, f2)
    Parallel(n_jobs=args.njobs)(delayed(proc_main)(jid, args) for jid in range(args.njobs))

if __name__=="__main__":
    main()
