from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter as arg_formatter
parser = ArgumentParser(formatter_class=arg_formatter)
parser.add_argument("outdir", help="path to output folder (will be created if necessary)", type=str)
parser.add_argument("cbf", type=str, help="path to cbf file for loading geometry")
parser.add_argument("--seed", default=0, help="random number seed", type=int)
parser.add_argument("--ngpu", default=1, type=int)
parser.add_argument("--nshot", default=15000, type=int)
parser.add_argument("--multiChance", type=float, default=0, help="number from 0-1, prob that a shot will be multi lattice")
#parser.add_argument("--multiSigma", type=float, default=0.1, help="in event of multi lattice shot, angular differences between orientations "
#                                                                  "will be generated using a Gaussian distribution with this sigma (degrees)")
parser.add_argument("--maxLat", type=int, default=3, help="in event of multi lattice shot, this many lattices will be simulated")
args = parser.parse_args()

from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD

import os
import h5py
import numpy as np
import time
from scipy.spatial.transform import Rotation

import dxtbx
from simtbx.diffBragg import utils

from resonet.sims.simulator import Simulator, reso2radius
from resonet.utils import eval_model


seeds = None
if COMM.rank == 0:
    np.random.seed(args.seed)
    seeds = np.random.permutation(999999)[:COMM.size]
seeds = COMM.bcast(seeds)
seed = seeds[COMM.rank]
np.random.seed(seed)


loader = dxtbx.load(args.cbf)
D = loader.get_detector()
D0 = utils.set_detector_thickness(D)
BEAM = loader.get_beam()
mask = loader.get_raw_data().as_numpy_array() >= 0

xdim,ydim = D0[0].get_image_size()

HS = Simulator(D0, BEAM)

detdist = abs(D0[0].get_origin()[2])
pixsize = D0[0].get_pixel_size()[0]
PIX_RADIUS_MAP = np.tan(2 * np.arcsin(HS.STOL * BEAM.get_wavelength())) * detdist / pixsize
dev = COMM.rank % args.ngpu

Nshot = len(np.array_split(np.arange(args.nshot), COMM.size)[COMM.rank])
multi_lattice = 2, .01
multi_lattice = None
all_param = []

if COMM.rank == 0:
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
COMM.barrier()
outname = os.path.join(args.outdir, "rank%d.h5" %COMM.rank)
with h5py.File(outname, "w") as out:
    out.create_dataset("pixel_radius_map", data=PIX_RADIUS_MAP)
    out.create_dataset("mask", data=mask)
    dset = out.create_dataset("images",
                              shape=(Nshot,) + (512,512),
                              dtype=np.float32)

    raw_dset = out.create_dataset("raw_images",
                              shape=(Nshot,) + tuple(mask.shape),
                              dtype=np.float32)

    rotMats = Rotation.random(Nshot).as_matrix()
    times = []
    for i_shot in range(Nshot):
        t = time.time()
        params, img = HS.simulate(rot_mat=rotMats[i_shot],
                                  multi_lattice_chance=args.multiChance,
                                  max_lat=args.maxLat,
                                  dev=dev)
        radius = reso2radius(params["reso"], D0, BEAM)
        all_param.append(
            [params["reso"], radius, params["multi_lattice"], params["ang_sigma"], params["num_lat"]])
        # TODO handle case for the Eiger!
        if xdim==2463:
            quad = eval_model.raw_img_to_tens_pil(img, mask)
        elif xdim==4096:
            quad = eval_model.raw_img_to_tens_mar(img, mask)
        else:
            quad = eval_model.raw_img_to_tens(img, mask)
        raw_dset[i_shot] = img
        dset[i_shot] = quad
        t = time.time()-t
        times.append(t)
        if COMM.rank == 0:
            print("Done with shot %d / %d (took %.4f sec)" % (i_shot+1, Nshot, t), flush=True)
    out.create_dataset("labels", data=all_param)
    if COMM.rank == 0:
        print("Done! Takes %.4f sec on average per image" % np.mean(times))
