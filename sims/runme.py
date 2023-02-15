from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter as arg_formatter
parser = ArgumentParser(formatter_class=arg_formatter)
parser.add_argument("outdir", help="path to output folder (will be created if necessary)", type=str)
parser.add_argument("cbf", type=str, help="path to cbf or mccd file for setting the simulation geometry")
parser.add_argument("--seed", default=None, help="random number seed. Default value of None will use int(time.time()) . Seed will be offset by MPI rank, so each rank always has a unique seed amongst all ranks.", type=int)
parser.add_argument("--ngpu", default=1, type=int, help="number of GPUs on machine")
parser.add_argument("--nshot", default=15000, type=int, help="number of shots to simulate")
parser.add_argument("--multiChance", type=float, default=0, help="number from 0-1. The probability that a shot will be multi lattice")
parser.add_argument("--maxLat", type=int, default=3, help="in event of multi lattice shot, this many lattices will be simulated")
parser.add_argument("--saveRaw", action="store_true", help="Save the raw diffraction images to the hdf5 files")
parser.add_argument("--mosMinMax", nargs=2, type=float, help="minium and maximum mosaic spread (mosaic spreads wil be drawn randomly, bound by these numbers). Default value of None will use MOS_MIN, MOS_MAX from paths_and_const.py")
parser.add_argument("--nmos", type=int, default=None, help="Number of mosaic blocks for sampling mosaicity. Default value of None will lead to ~1000 blocks per image (see choose_mos method in make_sims.py).")
parser.add_argument("--cpuMode", action="store_true", help="run computation on CPU (should specify small --nmos to speed up computation)")
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


# generate random seeds to each rank
seed_time = args.seed
if args.seed is None:
    if COMM.rank == 0:
        seed_time = int(time.time())
    seed_time = COMM.bcast(seed_time)
np.random.seed(seed_time+COMM.rank)

# load the geometry from provided image file
loader = dxtbx.load(args.cbf)
D = loader.get_detector()
# remove the sensor thickness portion of the geometry 
D0 = utils.set_detector_thickness(D)
BEAM = loader.get_beam()
# which pixel do not contain data
mask = loader.get_raw_data().as_numpy_array() > 0

# get the detector dimensions (used to determine detector model below)
xdim,ydim = D0[0].get_image_size()


# instantiate the simulator class
HS = Simulator(D0, BEAM, cuda=not args.cpuMode)

# sample-to-detector distance and pixel size
detdist = abs(D0[0].get_origin()[2])
pixsize = D0[0].get_pixel_size()[0]

# GPU device Id for this rank
dev = COMM.rank % args.ngpu

#  how many shots will this rank simulate
Nshot = len(np.array_split(np.arange(args.nshot), COMM.size)[COMM.rank])

# create output directory
if COMM.rank == 0:
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
COMM.barrier()

outname = os.path.join(args.outdir, "rank%d.h5" %COMM.rank)
all_param = []
with h5py.File(outname, "w") as out:
    out.create_dataset("mask", data=mask)
    dset = out.create_dataset("images",
                              shape=(Nshot,) + (512,512),
                              dtype=np.float32)

    if args.saveRaw:
        raw_dset = out.create_dataset("raw_images",
                                  shape=(Nshot,) + tuple(mask.shape),
                                  dtype=np.float32)

    # list of rotation matrices (length is Nshot)
    rotMats = Rotation.random(Nshot).as_matrix()
    times = []  # store processing times per shot
    for i_shot in range(Nshot):
        t = time.time()
        params, img = HS.simulate(rot_mat=rotMats[i_shot],
                                  multi_lattice_chance=args.multiChance,
                                  mos_min_max=args.mosMinMax,
                                  max_lat=args.maxLat,
                                  dev=dev, mos_dom_override=args.nmos)
        # at what pixel radius does this resolution corresond to
        radius = reso2radius(params["reso"], D0, BEAM)

        all_param.append(
            [params["reso"], radius, params["multi_lattice"], params["ang_sigma"], params["num_lat"]])

        # process the raw images according to detector model
        if xdim==2463:  # Pilatus 6M
            quad = eval_model.raw_img_to_tens_pil(img, mask)
        elif xdim==4096:  # Mar
            quad = eval_model.raw_img_to_tens_mar(img, mask)
        else:  # Eiger
            quad = eval_model.raw_img_to_tens(img, mask)
        if args.saveRaw:
            raw_dset[i_shot] = img
        dset[i_shot] = quad
        t = time.time()-t
        times.append(t)
        if COMM.rank == 0:
            print("Done with shot %d / %d (took %.4f sec)" % (i_shot+1, Nshot, t), flush=True)
    out.create_dataset("labels", data=all_param)
    if COMM.rank == 0:
        print("Done! Takes %.4f sec on average per image" % np.mean(times))
