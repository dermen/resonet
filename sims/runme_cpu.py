from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD

import os
import sys
import glob
import h5py
import numpy as np
import time
from scipy.spatial.transform import Rotation

import dxtbx
from simtbx.diffBragg import utils


from resonet.sims.simulator import Simulator, reso2radius


seeds = None
if COMM.rank == 0:
    np.random.seed(int(sys.argv[1]))
    seeds = np.random.permutation(999999)[:COMM.size]
seeds = COMM.bcast(seeds)
seed = seeds[COMM.rank]
np.random.seed(seed)


if __name__ == "__main__":
    from resonet.utils import eval_model

    fnames = glob.glob(os.path.join(sys.argv[4], "*.cbf"))
    assert fnames
    loader = dxtbx.load(fnames[0])
    D = loader.get_detector()
    D0 = utils.set_detector_thickness(D)
    BEAM = loader.get_beam()
    mask = loader.get_raw_data().as_numpy_array() >= 0

    xdim,ydim = D0[0].get_image_size()
    is_pil = xdim == 2463

    HS = Simulator(D0, BEAM)

    detdist = abs(D0[0].get_origin()[2])
    pixsize = D0[0].get_pixel_size()[0]
    PIX_RADIUS_MAP = np.tan(2 * np.arcsin(HS.STOL * BEAM.get_wavelength())) * detdist / pixsize
    NGPU = int(sys.argv[3])
    dev = COMM.rank % NGPU

    Nshot_tot = 15000
    Nshot = len(np.array_split(np.arange(Nshot_tot), COMM.size)[COMM.rank])
    multi_lattice = 2, .01
    multi_lattice = None
    all_param = []

    outdir = sys.argv[2]
    if COMM.rank == 0:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    COMM.barrier()
    outname = os.path.join(outdir, "rank%d.h5" %COMM.rank)
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
                                      ang_sigma=.1,
                                      multi_lattice_chance=0,
                                      dev=dev)
            radius = reso2radius(params["reso"], D0, BEAM)
            all_param.append(
                [params["reso"], radius, params["multi_lattice"]])
            # TODO handle case for the Eiger!
            if is_pil:
                quad = eval_model.raw_img_to_tens_pil(img, mask)
            else:
                quad = eval_model.raw_img_to_tens(img, mask)
            raw_dset[i_shot] = img
            dset[i_shot] = quad
            t = time.time()-t
            times.append(t)
            if COMM.rank==0:
                print("Done with shot %d / %d (took %.4f sec)" % (i_shot+1, Nshot, t), flush=True)
        out.create_dataset("labels", data=all_param)
        if COMM.rank == 0:
            print("Done! Takes %.4f sec on average per image" % np.mean(times))
