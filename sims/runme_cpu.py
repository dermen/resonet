
from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD
import glob
import h5py
import dxtbx
import numpy as np
from simtbx.diffBragg import utils
from scipy.spatial.transform import Rotation
import time
import make_crystal
from simtbx.nanoBragg import sim_data
from dials.array_family import flex

# load a detector and beam model
import make_sims
import paths_and_const

np.random.seed(0)


class Simulator:

    def __init__(self, DET, BEAM):
        self.DET = DET
        self.BEAM = BEAM
        print("sim bg")
        # air and water background:
        self.air_and_water = make_sims.get_background(self.DET, self.BEAM)
        # stol of every pixel:
        self.STOL = make_sims.get_theta_map(self.DET, self.BEAM)

        self.nb_beam = make_crystal.load_beam(self.BEAM,
                                      divergence=paths_and_const.DIVERGENCE_MRAD / 1e3 * 180 / np.pi)


    def simulate(self, rot_mat=None, multi_lattice=False):
        """

        :param rot_mat: nominal rotation matrix
        :param multi_lattice: a 2-tuple, where first element is the number of additional lattices to simulate
            and the second element is the stdev. of a gaussian (in degrees) used to determine perturbation angles
        :return:
        """
        pdb_name = make_sims.choose_pdb()
        C = make_crystal.load_crystal(pdb_name, rot_mat, use_p1=True)
        mos_spread, mos_dom = make_sims.choose_mos()
        print("Simulating for %s" % pdb_name)
        print(mos_spread, mos_dom)
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

        print("sim spots!")
        S.D.add_nanoBragg_spots_cuda()
        spots = S.D.raw_pixels.as_numpy_array()
        if multi_lattice is not None:
            num_lat, ang_sigma = multi_lattice
            mats = Rotation.random(num_lat).as_matrix()
            vecs = np.dot(np.array([1, 0, 0])[None], mats)[0]
            angs = np.random.normal(0, ang_sigma, num_lat)
            scaled_vecs = vecs*angs[:, None]
            rot_mats = Rotation.from_rotvec(scaled_vecs).as_matrix()

            nominal_crystal = S.crystal.dxtbx_crystal
            Umat = np.reshape(nominal_crystal.get_U(), (3, 3))
            for i_p, perturb in enumerate(rot_mats):
                print("additional multi lattice sim %d" % (i_p+1))
                Umat_p = np.dot(perturb, Umat)
                nominal_crystal.set_U(tuple(Umat_p.ravel()))
                S.D.Amatrix = sim_data.Amatrix_dials2nanoBragg(nominal_crystal)

                S.D.raw_pixels *= 0
                S.D.add_nanoBragg_spots_cuda()
                spots += S.D.raw_pixels.as_numpy_array()
            spots /= (num_lat+1)

        print("sim random bg")
        plastic_stol = np.random.choice(paths_and_const.RANDOM_STOLS)
        plastic = make_sims.random_bg(self.DET, self.BEAM, plastic_stol)

        reso, Bfac_img = make_sims.get_Bfac_img(self.STOL)

        img = spots*Bfac_img*paths_and_const.VOL + self.air_and_water + plastic

        make_sims.set_noise(S.D)

        S.D.raw_pixels = flex.double(img.ravel())
        S.D.add_noise()
        noise_img = S.D.raw_pixels.as_numpy_array().reshape(img.shape)

        return noise_img




if __name__ == "__main__":
    fnames = glob.glob("/mnt/data/s2/blstaff/SOLTIS/AI_PREDICTION/3.15A/*cbf")
    loader = dxtbx.load(fnames[0])
    D = loader.get_detector()
    D0 = utils.set_detector_thickness(D)
    BEAM = loader.get_beam()
    mask = loader.get_raw_data().as_numpy_array() < 0

    HS = Simulator(D0, BEAM)
    Nshot = 2
    multi_lattice = 2,.01
    multi_lattice = None

    with h5py.File("holtonator_tester_rank%d_HiSym2.h5" % COMM.rank, "w") as out:
        out.create_dataset("mask", data=mask)
        dset = out.create_dataset("images",
                                  shape=(Nshot,) + tuple(mask.shape),
                                  dtype=np.float32)

        rotMats = Rotation.random(Nshot).as_matrix()
        times = []
        for i_shot in range(Nshot):
            if i_shot % COMM.size != COMM.rank :
                continue
            t = time.time()
            img = HS.simulate(rot_mat=rotMats[i_shot], multi_lattice=multi_lattice)
            dset[i_shot] = img
            t = time.time()-t
            times.append(t)
            print("Done with shot %d / %d (took %.4f sec)" % (i_shot+1, Nshot, t))
        print("Done! Takes %.4f sec on average per image" % np.mean(times))

