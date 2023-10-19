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
import time

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("datadir", help="Path to folder containing validation mccd or cbf images.", type=str)
parser.add_argument("modelname", help="path to the .nn model", type=str)
parser.add_argument("outfile", type=str, help="name of the output file that will be created")
parser.add_argument("--arch", type=str, default="res34",
                    help="architecture of model (default: res34)")
parser.add_argument("--gain", default=0.57, type=float, help="adu to photon conversion factor (default=0.57)")
parser.add_argument("--figdir", default=None, type=str,
                    help="A directory for PNG files to be written to. "
                         "Default: tempfigs_X where X is a string representing resolution")
parser.add_argument("--display", action="store_true")
parser.add_argument("--gpus", action="store_true")
args = parser.parse_args()

model = load_model(args.modelname, args.arch)

fnames = glob.glob(args.datadir + "/*_[0-9]_*mccd")
if not fnames:
    fnames = glob.glob(args.datadir + "/*_[0-9]_*cbf")


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


#fnames = sanitize_inputs(fnames)
#assert fnames

loader = dxtbx.load(fnames[0])

mask = loader.get_raw_data().as_numpy_array() > 0

xdim, ydim = loader.get_detector()[0].get_image_size()
is_pil = xdim==2463

Nf = len(fnames)
if COMM.rank==0:
    print("Found %d fnames" % Nf)

#figdir = args.figdir
#if figdir is None:
#    figdir = args.outfile + ".resultfigs"
#if COMM.rank==0 and not os.path.exists(figdir):
#    os.makedirs(figdir)
from resonet.utils import mpi
dev = "cpu"
if args.gpus:
    gpu_id = mpi.get_gpu_id_mem(1)
    dev = "cuda:%d" % gpu_id
    model = model.to(dev)

COMM.barrier()
rank_fnames = []
rank_fignames = []
raw_preds = []
labels = []
read_times = []
predict_times = []
ds_times = []
from resonet.utils import eval_model

factor = 2 if is_pil else 4
maxpool = torch.nn.MaxPool2d(factor, factor)
kwargs = {}
kwargs["dev"] = dev
kwargs["maxpool"] = maxpool
kwargs["ds_fact"] = factor
kwargs["cent"] = xdim / 2., ydim / 2.

for i_f, f in enumerate(fnames):
    if i_f % COMM.size != COMM.rank: continue
    t = time.time()
    try:
        loader = dxtbx.load(f)
    except:
        continue
    img = loader.get_raw_data().as_numpy_array().astype(np.float32) / args.gain
    t = time.time()-t
    read_times.append(t)
    t2 = time.time()
    tens = eval_model.to_tens(img, mask, quad="A", **kwargs)
    #if is_pil:
    #    tens = raw_img_to_tens_pil(img, mask)
    #else:
    #    if xdim == 4096:  # is MAR (note, depends on binning
    #        tens = raw_img_to_tens_mar(img, mask)
    #    else:
    #        tens = raw_img_to_tens(img, mask)
    t2 = time.time()-t2
    ds_times.append(t2)

    t3 = time.time()
    tens = tens.to(dev)
    assert dev.startswith("cuda")
    raw_prediction = torch.sigmoid(model(tens))
    is_multi = int(torch.round(raw_prediction).item())
    t3 = time.time()-t3
    predict_times.append(t3)
    raw_preds.append(raw_prediction.item())
    labels.append(is_multi)
    if COMM.rank==0:
        print("Done with image %d / %d (raw=%.2f) (read: %.4fsec , dwnsamp: %.4fsec, predict: %.4fsec)" % (i_f, len(fnames), raw_prediction, t,t2,t3), flush=True)
    
    rank_fnames.append(f)
    #plt.cla()
    #tens_im = tens.numpy()[0,0,:512, :512]
    #m = tens_im.mean()
    #s = tens_im.std()
    #vmax = m+3*s
    #vmin = m-2
    #plt.imshow(tens.numpy()[0, 0, :512, :512], vmax=vmax, vmin=vmin)
    tag = "multi" if is_multi else "single"
    #plt.title("%s: %s" % (os.path.basename(f), tag))
    #figname = "%s/GAIN%.4f_%05d.png" % (figdir,args.gain, i_f)
    #plt.savefig(figname, dpi=150)
    #rank_fignames.append(figname)

    #if COMM.rank==0:
    #    plt.draw()
    #    plt.pause(0.2)

read_times = COMM.reduce(read_times)
ds_times = COMM.reduce(ds_times)
predict_times = COMM.reduce(predict_times)
raw_preds = COMM.reduce(raw_preds)
fnames = COMM.reduce(rank_fnames)
rank_fignames = COMM.reduce(rank_fignames)
labels = COMM.reduce(labels)

if COMM.rank==0:
    slab = np.sum(labels)
    nlab = len(labels)
    fraction = slab / nlab * 100
    tread = np.mean(read_times)
    tpred = np.mean(predict_times)
    tds = np.mean(ds_times)
    s = "%s;%s %.2f %% of shots (%d/%d) contained multiple lattices (%.4f sec/read %.4f sec/predict, %.4f sec/dwnsamp)" % (args.datadir, args.modelname, fraction, slab, nlab, tread, tpred, tds)
    print(s)
    np.savez(args.outfile, raw=raw_preds, fnames=fnames,fignames=rank_fignames, labels=labels, results_string=s, read_times=read_times, predict_times=predict_times)

