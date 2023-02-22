# coding: utf-8

from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD

import numpy as np
import dxtbx
import glob
import sys
import os
from resonet.utils.eval_model import load_model, raw_img_to_tens_pil, raw_img_to_tens, raw_img_to_tens_mar
from resonet.sims.simulator import reso2radius
import pylab as plt
import torch

MODEL_FNAME = sys.argv[1]
outfile = sys.argv[2]
GAIN = float(sys.argv[3])  # 0.57 from dials.estimate_gain
arch = "res34"

model = load_model(MODEL_FNAME, arch)

dirname = "/global/cscratch1/sd/dermen/baxter_Peters_reduced/reduced/"
fnames = glob.glob(dirname + "/*mccd") 

loader = dxtbx.load(fnames[0])

mask = loader.get_raw_data().as_numpy_array() > 0
Nf = len(fnames)
if COMM.rank==0:
    print("Found %d fnames" % Nf)
figdir = "baxter_figs"
if not os.path.exists(figdir) and COMM.rank==0:
    os.makedirs(figdir)
COMM.barrier()
rank_fnames = []
rank_fignames = []
raw_preds = []
labels = []
for i_f, f in enumerate(fnames):
    if i_f % COMM.size != COMM.rank: continue
    loader = dxtbx.load(f)
    img = loader.get_raw_data().as_numpy_array() / GAIN
    tens = raw_img_to_tens_mar(img, mask)

    raw_prediction = torch.sigmoid(model(tens))
    is_multi = int(torch.round(raw_prediction).item())
    raw_preds.append(raw_prediction.item())
    labels.append(is_multi)
    
    rank_fnames.append(f)
    plt.cla()
    plt.imshow(tens.numpy()[0, 0, :512, :512], vmax=10)
    tag = "multi" if is_multi else "single"
    plt.title("%s: %s" % (os.path.basename(f), tag))
    figname = "%s/GAIN%.4f_%05d.png" % (figdir,GAIN, i_f)
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
    np.savez(outfile, raw=raw_preds, fnames=fnames,fignames=rank_fignames, labels=labels)

