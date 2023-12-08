
import h5py
import glob
import numpy as np
from joblib import Parallel, delayed

"""
Example usage: python compress.py folder1 fodler2 10
The above command uses 10 processes to compress the training files in folders folder1 and fodler2
The files should compress considerably.
"""


def proc_main(jid, dirname, numJob):

    fnames = glob.glob(dirname + "/rank*.h5")

    for i_f, f in enumerate(fnames):
        if i_f % numJob != jid:
            continue
        fnew = f.replace("rank", "compressed")
        h = h5py.File(f, "r")
        imgs = h['images']
        labs = h['labels'][()]
        geom = h['geom'][()]

        with h5py.File(fnew, "w") as hnew:
            dset_im = hnew.create_dataset('images',
                    dtype=np.uint16,
                    compression="gzip",
                    shape=h['images'].shape,
                    compression_opts=4)
            for i_img in range(imgs.shape[0]):
                if i_img % 10==0: # and jid==0:
                    print(f"done with file {i_f+1}/{len(fnames)} shot {i_img}/{imgs.shape[0]}")
                im = imgs[i_img].astype(np.uint16)
                dset_im[i_img] = im

            dset_lab = hnew.create_dataset("labels", data=labs.astype(np.float32),
                compression="gzip", compression_opts=4)
            dset_lab.attrs['names'] = h['labels'].attrs['names']
            dset_lab.attrs['pdbmap'] = h['labels'].attrs['pdbmap']

            dset_geom = hnew.create_dataset("geom", data=geom.astype(np.float32),
                compression="gzip", compression_opts=4)
            dset_geom.attrs['names'] = h['geom'].attrs['names']


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("dirname", type=str, help="training data output folder")
    parser.add_argument("nj", type=int, help="number of parallel jobs to run")
    args = parser.parse_args()
    Parallel(args.nj)(delayed(proc_main)(j, args.dirname, args.nj) for j in range(args.nj))


if __name__=="__main__":
    main()

