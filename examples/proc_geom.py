# coding: utf-8

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("datadir", help="Path to folder containing validation mccd or cbf images. "
                                    "The dirname is usually of the format,  e.g. 1.90A", type=str)
parser.add_argument("modelname", help="path to the .nn model", type=str)
parser.add_argument("outfile", type=str, help="name of the output file that will be created")
parser.add_argument("--arch", type=str, choices=["res50", "res18", "res34", "le"], default="res50",
                    help="architecture of model (default: res50)")
parser.add_argument("--geom", action="store_true")
parser.add_argument("--predictor", type=str, choices=["rad", "one_over_rad", "res", "one_over_res"],
                    default="one_over_rad")
parser.add_argument("--figdir", default=None, type=str,
                    help="A directory for PNG files to be written to. "
                         "Default: tempfigs_X where X is a string representing resolution")
parser.add_argument("--maskFile", type=str, default=None)
args = parser.parse_args()

from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD

import numpy as np
import dxtbx
import glob
import re
import os
from resonet.utils.eval_model import load_model, raw_img_to_tens_pil, raw_img_to_tens
from resonet.sims.simulator import reso2radius
import pylab as plt
from scipy.ndimage import binary_dilation
import torch


real_data_dirname = args.datadir
assert os.path.isdir(real_data_dirname)
MODEL_FNAME = args.modelname
model = load_model(MODEL_FNAME, arch=args.arch)

fnames = glob.glob(real_data_dirname + "/*[0-9].cbf")
if not fnames:
    fnames = glob.glob(real_data_dirname + "/*[0-9].mccd")
    # TODO add tensor conversion for MCCD files...

def sanitize_inputs(fnames):
    good_fnames = []
    for i,f in enumerate(fnames):
        if i % COMM.size != COMM.rank:
            continue
        try:
            loader = dxtbx.load(f)
            good_fnames.append(f)
        except KeyError:
            continue

        if COMM.rank==0:
            print("Verifying shot %d/ %d" % (i, len(fnames)))
    good_fnames = COMM.bcast(COMM.reduce(good_fnames))
    return good_fnames


#fnames = sanitize_inputs(fnames)
assert fnames


def res_from_name(name):
    res = re.findall("[0-9]\.[0-9]+A", name)
    assert len(res)==1
    res = float(res[0].split("A")[0])
    return res

target_res = res_from_name(real_data_dirname)

loader = dxtbx.load(fnames[0])
B = loader.get_beam()
D = loader.get_detector()
detdist = abs(D[0].get_distance())
wavelen = B.get_wavelength()
pixsize = D[0].get_pixel_size()[0]

#imgs = []
#for i,f in enumerate(fnames[:20]):
#    if i% COMM.size != COMM.rank:
#        continue
#    if COMM.rank==0:
#        print("Inspecting for hot pixels... %d/%d" %(i, 100), flush=True)
#    img = dxtbx.load(f).get_raw_data().as_numpy_array().astype(np.float32)
#    img[img > 1e4] = 0
#    imgs.append(img)
#imgs = COMM.reduce(imgs)
#hotpix = None
#if COMM.rank==0:
#    print("median...", flush=True)
#    img_med =np.median(imgs, 0)
#    hotpix = img_med > 1e2
#hotpix = COMM.bcast(hotpix)


xdim, ydim = D[0].get_image_size()

is_pil = xdim==2463

if args.maskFile is None:
    mask = loader.get_raw_data().as_numpy_array() >= 0
    mask = ~binary_dilation(~mask, iterations=2)
    beamstop_rad = 10
    Y,X = np.indices((ydim, xdim))
    R = np.sqrt((X-xdim/2.)**2 + (Y-ydim/2.)**2)
    out_of_beamstop = R > beamstop_rad
    mask = np.logical_and(mask, out_of_beamstop)
else:
    mask = np.load(args.maskFile)

#mask = np.logical_and(mask, ~hotpix)

Nf = len(fnames)
if COMM.rank==0:
    print("Found %d fnames" % Nf)
factor = 2 if is_pil else 4
target_rad = reso2radius(target_res, DET=D, BEAM=B) / factor
if COMM.rank==0:
    print("Target res: %fA" % target_res)

figdir = args.figdir
if figdir is None:
    figdir = args.outfile + ".tempfigs"
if COMM.rank==0 and not os.path.exists(figdir):
    os.makedirs(figdir)
COMM.barrier()

geom = torch.tensor([[detdist, pixsize, wavelen, xdim, ydim]])
rank_fnames = []
rank_fignames = []
all_res = []
rads = []
for i_f, f in enumerate(fnames):
    if i_f % COMM.size != COMM.rank: continue
    loader = dxtbx.load(f)
    img = loader.get_raw_data().as_numpy_array()
    if is_pil:
        tens = raw_img_to_tens_pil(img, mask)
    else:
        tens = raw_img_to_tens(img, mask)

    inputs = (tens,)
    if args.geom:
        geom = torch.tensor([[detdist, pixsize, wavelen, xdim, ydim]])
        inputs = (tens, geom)
    pred = model(*inputs).item()

    if args.predictor in ["one_over_res", "res"]:
        if args.predictor =="res":
            res = pred
        else:
            res = 1/pred
        radius = reso2radius(res, D, B) / factor
    else: # args.predictor in ["rad", "one_over_rad"]:
        if args.predictor == "rad":
            radius = pred
        else:
            radius = 1/pred
        theta = 0.5*np.arctan(radius*factor*pixsize/detdist)
        res = 0.5*wavelen/np.sin(theta)

    print(radius, target_rad)
    all_res.append(res)
    rads.append(radius)
    rank_fnames.append(f)
    plt.cla()
    plt.imshow(tens.numpy()[0, 0, :512, :512], vmax=10)
    plt.gca().add_patch(plt.Circle(xy=(0,0), radius=radius, ec='r', ls='--', fc='none' ))
    plt.gca().add_patch(plt.Circle(xy=(0,0), radius=target_rad, ec='w', ls='--', fc='none' ))
    plt.title("%.2fA: %s" % (target_res, os.path.basename(f)))
    figname = os.path.join( figdir, "%.2fA_%05d.png" % (target_res, i_f))
    plt.savefig(figname, dpi=150)
    rank_fignames.append(figname)

    if COMM.rank==0:
        plt.draw()
        plt.pause(0.2)

rads = COMM.reduce(rads)
res = COMM.reduce(all_res)
fnames = COMM.reduce(rank_fnames)
rank_fignames = COMM.reduce(rank_fignames)

if COMM.rank==0:
    res = np.array(res)
    order = np.argsort(rads)[::-1]
    ordered_figs = [rank_fignames[i] for i in order]
    for i, f in enumerate(ordered_figs):
        new_f = os.path.join(figdir, "sorted_%05d.png" % i)
        if os.path.exists(new_f):
            os.remove(new_f)
        os.symlink(os.path.abspath(f), new_f)

    perc10 = np.percentile(res,10)
    max_top10 = res[res <= perc10].mean()
    s = real_data_dirname + "Res: %.4f +- %.4f (Angstrom). highest= %.4fA . MeanHighest10perc=%.4fA (detdist=%.2f mm)" \
        % (np.mean(res), np.std(res), np.min(res), max_top10, detdist)
    print(s)
    np.savez(args.outfile, rads=rads, fnames=fnames, pixsize=pixsize, detdist=detdist, wavelen=wavelen, res=res,
             result_string=s, fignames=rank_fignames, factor=factor, target_rad=target_rad, target_res=target_res)
    o = args.outfile
    if not o.endswith(".npz"):
        o = o + ".npz"
    print("Saved file %s" % o)
    print("Done.")
