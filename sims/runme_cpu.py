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
from simtbx.nanoBragg import sim_data
from dials.array_family import flex
from resonet.sims import make_crystal, make_sims, paths_and_const

seeds = None
if COMM.rank == 0:
    np.random.seed(int(sys.argv[1]))
    seeds = np.random.permutation(999999)[:COMM.size]
seeds = COMM.bcast(seeds)
seed = seeds[COMM.rank]
np.random.seed(seed)


class Simulator:

    def __init__(self, DET, BEAM):
        self.DET = DET
        self.BEAM = BEAM
        # air and water background:
        self.air_and_water = make_sims.get_background(self.DET, self.BEAM)
        # stol of every pixel:
        self.STOL = make_sims.get_theta_map(self.DET, self.BEAM)

        self.nb_beam = make_crystal.load_beam(self.BEAM,
                                      divergence=paths_and_const.DIVERGENCE_MRAD / 1e3 * 180 / np.pi)

    def simulate(self, rot_mat=None, multi_lattice_chance=0, ang_sigma=0.01, num_lat=2):
        """

        :param rot_mat: nominal rotation matrix
        :param multi_lattice_chance:
        :return:
        """
        pdb_name = make_sims.choose_pdb()
        C = make_crystal.load_crystal(pdb_name, rot_mat, use_p1=False)
        mos_spread, mos_dom = make_sims.choose_mos()
        #print("Simulating for %s" % pdb_name)
        #print(mos_spread, mos_dom)
        C.mos_spread_deg = mos_spread
        n_mos = mos_dom//2
        if n_mos % 2 == 1:
            n_mos += 1
        C.n_mos_domains = n_mos

        S = sim_data.SimData()
        S.crystal = C
        S.beam = self.nb_beam
        S.detector = D0
        S.instantiate_nanoBragg(oversample=1)

        #S.D.show_params()

        #print("sim spots!")
        S.D.add_nanoBragg_spots_cuda()
        spots = S.D.raw_pixels.as_numpy_array()
        use_multi = np.random.random() < multi_lattice_chance
        if use_multi:
            mats = Rotation.random(num_lat).as_matrix()
            vecs = np.dot(np.array([1, 0, 0])[None], mats)[0]
            angs = np.random.normal(0, ang_sigma, num_lat)
            scaled_vecs = vecs*angs[:, None]
            rot_mats = Rotation.from_rotvec(scaled_vecs).as_matrix()

            nominal_crystal = S.crystal.dxtbx_crystal
            Umat = np.reshape(nominal_crystal.get_U(), (3, 3))
            for i_p, perturb in enumerate(rot_mats):
                #print("additional multi lattice sim %d" % (i_p+1))
                Umat_p = np.dot(perturb, Umat)
                nominal_crystal.set_U(tuple(Umat_p.ravel()))
                S.D.Amatrix = sim_data.Amatrix_dials2nanoBragg(nominal_crystal)

                S.D.raw_pixels *= 0
                S.D.add_nanoBragg_spots_cuda()
                spots += S.D.raw_pixels.as_numpy_array()
            spots /= (num_lat+1)

        #print("sim random bg")
        plastic_stol = np.random.choice(paths_and_const.RANDOM_STOLS)
        plastic = make_sims.random_bg(self.DET, self.BEAM, plastic_stol)

        reso, Bfac_img = make_sims.get_Bfac_img(self.STOL)

        img = spots*Bfac_img*paths_and_const.VOL + self.air_and_water + plastic

        make_sims.set_noise(S.D)

        S.D.raw_pixels = flex.double(img.ravel())
        S.D.add_noise()
        noise_img = S.D.raw_pixels.as_numpy_array().reshape(img.shape)

        param_dict = {"reso": reso,
                      "multi_lattice": use_multi,
                      "ang_sigma": ang_sigma,
                      "num_lat": num_lat}

        S.D.free_all()

        return param_dict, noise_img


def reso2radius(reso, DET, BEAM):
    """

    :param reso: resolution of a position on camera in Angstrom
    :param DET: dxtbx detector model
    :param BEAM: dxtbx beam model
    :return: the resolution converted to pixel radii (distance from beam center)
    """
    wavelen = BEAM.get_wavelength()
    detdist = abs(DET[0].get_origin()[2])  # approximate for tilted cameras!
    pixsize = DET[0].get_pixel_size()[0]  # assumes square pixel

    theta = np.arcsin(wavelen/2/reso)
    rad = np.tan(2*theta) * detdist/pixsize
    return rad


if __name__ == "__main__":
    from resonet.utils import eval_model

    fnames = glob.glob("geoms/5.40A/*cbf")
    loader = dxtbx.load(fnames[0])
    D = loader.get_detector()
    D0 = utils.set_detector_thickness(D)
    BEAM = loader.get_beam()
    mask = loader.get_raw_data().as_numpy_array() >= 0

    HS = Simulator(D0, BEAM)

    detdist = abs(D0[0].get_origin()[2])
    pixsize = D0[0].get_pixel_size()[0]
    PIX_RADIUS_MAP = np.tan(2 * np.arcsin(HS.STOL * BEAM.get_wavelength())) * detdist / pixsize

    Nshot_tot = 20000
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
            #if i_shot % COMM.size != COMM.rank :
            #    continue
            t = time.time()
            params, img = HS.simulate(rot_mat=rotMats[i_shot],
                                      ang_sigma=.1,
                                      multi_lattice_chance=0)
            radius = reso2radius(params["reso"], D0, BEAM)
            all_param.append(
                [radius, params["multi_lattice"]])
            quad = eval_model.raw_img_to_tens_pil(img, mask)
            raw_dset[i_shot] = img
            dset[i_shot] = quad
            t = time.time()-t
            times.append(t)
            if COMM.rank==0:
                print("Done with shot %d / %d (took %.4f sec)" % (i_shot+1, Nshot, t), flush=True)
        out.create_dataset("labels", data=all_param)
        if COMM.rank == 0:
            print("Done! Takes %.4f sec on average per image" % np.mean(times))
