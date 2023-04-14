# coding: utf-8

from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD

import numpy as np
import dxtbx
import glob
import os
from resonet.utils.eval_model import load_model, raw_img_to_tens_pil, raw_img_to_tens, raw_img_to_tens_mar
import pylab as plt
import torch

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("datadir", help="Path to folder containing validation mccd or cbf images. ", type=str)
parser.add_argument("modelname", help="path to the .nn model", type=str)
parser.add_argument("outfile", type=str, help="name of the output file that will be created")
parser.add_argument("--arch", type=str, default="res34",
                    help="architecture of model (default: res34)")
parser.add_argument("--gain", default=1, type=float, help="adu to photon conversion factor")
parser.add_argument("--figdir", default=None, type=str,
                    help="A directory for PNG files to be written to. "
                         "Default: tempfigs_X where X is a string representing resolution")
args = parser.parse_args()

model = load_model(args.modelname, args.arch)

fnames = glob.glob(args.datadir + "/*mccd")
if not fnames:
    fnames = glob.glob(args.datadir + "/*cbf")


def sanitize_inputs(fnames):
    good_fnames = []
    for i,f in enumerate(fnames):
        if i % COMM.size != COMM.rank:
            continue
        try:
            _ = dxtbx.load(f)
            good_fnames.append(f)
        except KeyError:
            continue

        if COMM.rank==0:
            print("Verifying shot %d/ %d" % (i, len(fnames)))
    good_fnames = COMM.bcast(COMM.reduce(good_fnames))
    return good_fnames


fnames = sanitize_inputs(fnames)
assert fnames

loader = dxtbx.load(fnames[0])

mask = loader.get_raw_data().as_numpy_array() > 0

xdim, ydim = loader.get_detector()[0].get_image_size()
is_pil = xdim==2463

Nf = len(fnames)
if COMM.rank==0:
    print("Found %d fnames" % Nf)

figdir = args.figdir
if figdir is None:
    figdir = args.outfile + ".resultfigs"
if COMM.rank==0 and not os.path.exists(figdir):
    os.makedirs(figdir)
COMM.barrier()
rank_fnames = []
rank_fignames = []
raw_preds = []
labels = []
for i_f, f in enumerate(fnames):
    if i_f % COMM.size != COMM.rank: continue
    loader = dxtbx.load(f)
    img = loader.get_raw_data().as_numpy_array() / args.gain
    if is_pil:
        tens = raw_img_to_tens_pil(img, mask)
    else:
        if xdim == 4096:  # is MAR (note, depends on binning
            tens = raw_img_to_tens_mar(img, mask)
        else:
            tens = raw_img_to_tens(img, mask)

    raw_prediction = torch.sigmoid(model(tens))
    is_multi = int(torch.round(raw_prediction).item())
    raw_preds.append(raw_prediction.item())
    labels.append(is_multi)
    
    rank_fnames.append(f)
    plt.cla()
    tens_im = tens.numpy()[0,0,:512, :512]
    m = tens_im.mean()
    s = tens_im.std()
    vmax = m+3*s
    vmin = m-2
    plt.imshow(tens.numpy()[0, 0, :512, :512], vmax=vmax, vmin=vmin)
    tag = "multi" if is_multi else "single"
    plt.title("%s: %s" % (os.path.basename(f), tag))
    figname = "%s/GAIN%.4f_%05d.png" % (figdir,args.gain, i_f)
    plt.savefig(figname, dpi=150)
    rank_fignames.append(figname)

    if COMM.rank==0:
        plt.draw()
        plt.pause(0.2)

raw_preds = COMM.reduce(raw_preds)
fnames = COMM.reduce(rank_fnames)
rank_fignames = COMM.reduce(rank_fignames)
labels = COMM.reduce(labels)

if COMM.rank==0:
    fraction = np.sum(labels) / len(labels) * 100
    print("%.2f %% of shots contained multiple lattices" % fraction)
    np.savez(args.outfile, raw=raw_preds, fnames=fnames,fignames=rank_fignames, labels=labels)

