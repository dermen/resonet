# coding: utf-8

from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD

import numpy as np
import dxtbx
import glob
import sys
import os
from resonet.utils.eval_model import load_model, raw_img_to_tens_pil, raw_img_to_tens
from resonet.sims.simulator import reso2radius
import pylab as plt


real_data_dirname = sys.argv[1]
assert os.path.isdir(real_data_dirname)
MODEL_FNAME = sys.argv[2]
model = load_model(MODEL_FNAME)
outfile = sys.argv[3]

fnames = glob.glob(real_data_dirname + "/*cbf") #"/mnt/data/s2/blstaff/SOLTIS/AI_PREDICTION/1.87A/*cbf")

import re
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

imgs = []
for i,f in enumerate(fnames[:100]):
    if i% COMM.size != COMM.rank:
        continue
    if COMM.rank==0:
        print("Inspecting for hot pixels... %d/%d" %(i, 100), flush=True)
    img = dxtbx.load(f).get_raw_data().as_numpy_array()
    imgs.append(img)
imgs = COMM.reduce(imgs)
hotpix = None
if COMM.rank==0:
    print("median...", flush=True)
    img_med =np.median(imgs, 0)
    hotpix = img_med > 1e3
hotpix = COMM.bcast(hotpix)


xdim, ydim = D[0].get_image_size()

is_pil = xdim==2463

#if is_pil:
#    mask = utils.load_mask("pilatus.mask")
#else:
#    mask = utils.load_mask("1.87A.mask")
#
# Mask used during training:
mask = loader.get_raw_data().as_numpy_array() >= 0
mask = np.logical_and(mask, ~hotpix)


Nf = len(fnames)
if COMM.rank==0:
    print("Found %d fnames" % Nf)
rads = []
factor = 2 if is_pil else 4
target_rad = reso2radius(target_res, DET=D, BEAM=B)
if COMM.rank==0:
    print("Target res: %fA" % target_res)
rank_fnames = []
rank_fignames = []
for i_f, f in enumerate(fnames):
    if i_f % COMM.size != COMM.rank: continue
    loader = dxtbx.load(f)
    img = loader.get_raw_data().as_numpy_array()
    if is_pil:
        tens = raw_img_to_tens_pil(img, mask)
    else:
        tens = raw_img_to_tens(img, mask)
    radius = model(tens).item()
    rads.append(radius)
    rank_fnames.append(f)
    plt.cla()
    plt.imshow(tens.numpy()[0, 0, :512, :512], vmax=10)
    plt.gca().add_patch(plt.Circle(xy=(0,0), radius=radius/factor, ec='r', ls='--', fc='none' ))
    plt.gca().add_patch(plt.Circle(xy=(0,0), radius=target_rad/factor, ec='w', ls='--', fc='none' ))
    plt.title("%.2fA: %s" % (target_res, os.path.basename(f)))
    figname = "results_figs/%.2fA_%05d.png" % (target_res, i_f)
    plt.savefig(figname, dpi=150)
    rank_fignames.append(figname)

    if COMM.rank==0:
        plt.draw()
        plt.pause(0.2)

rads = COMM.reduce(rads)
fnames = COMM.reduce(rank_fnames)
rank_fignames = COMM.reduce(rank_fignames)

if COMM.rank==0:
    order = np.argsort(rads)[::-1]
    ordered_figs = [rank_fignames[i] for i in order]
    tempdir="tempfigs_%.2fA" % target_res
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
    for i,f in enumerate(ordered_figs):
        new_f = os.path.join(tempdir, "%05d.png" % i)
        if os.path.exists(new_f):
            os.remove(new_f)
        os.symlink(os.path.abspath(f), new_f)

    rads = np.array(rads)
    factor = 1
    print("pixel scale=%d"%factor)
    theta = np.arctan(rads*pixsize*factor/detdist)
    q = 2/wavelen*np.sin(theta/2.)
    res = 1/q
    perc10 = np.percentile(res,10)
    max_top10 = res[res <= perc10].mean()
    s = real_data_dirname + "Res: %.4f +- %.4f (Angstrom). highest= %.4fA . MeanHighest10perc=%.4fA (detdist=%.2f mm)" % (np.mean(res), np.std(res), np.min(res), max_top10, detdist)
    print(s)
    np.savez(outfile, rads=rads, fnames=fnames, pixsize=pixsize, detdist=detdist, wavelen=wavelen, res=res, result_string=s, fignames=rank_fignames)

