
import os
import numpy as np
from scipy.spatial.transform import Rotation

from simtbx.nanoBragg import sim_data
from dials.array_family import flex


from resonet.sims import make_sims, make_crystal, paths_and_const


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

    def simulate(self, rot_mat=None, multi_lattice_chance=0, ang_sigma=0.01, num_lat=2, mos_tuple=None,
                 pdb_name=None, plastic_stol=None, dev=0):
        """

        :param rot_mat:
        :param multi_lattice_chance:
        :param ang_sigma:
        :param num_lat:
        :param mos_tuple:
        :param pdb_name:
        :param plastic_stol:
        :param dev: device id
        :return:
        """
        if pdb_name is None:
            pdb_name = make_sims.choose_pdb()
        else:
            assert os.path.exists(pdb_name)
            assert os.path.isdir(pdb_name)
        C = make_crystal.load_crystal(pdb_name, rot_mat)
        if mos_tuple is None:
            mos_spread, mos_dom = make_sims.choose_mos()
        else:
            mos_spread, mos_dom = mos_tuple
        C.mos_spread_deg = mos_spread
        n_mos = mos_dom//2
        if n_mos % 2 == 1:
            n_mos += 1
        C.n_mos_domains = n_mos

        S = sim_data.SimData()
        S.crystal = C
        S.beam = self.nb_beam
        S.detector = self.DET
        S.instantiate_nanoBragg(oversample=1)

        #S.D.show_params()

        S.D.device_Id = dev
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
        if plastic_stol is None:
            plastic_stol = np.random.choice(paths_and_const.RANDOM_STOLS)
        else:
            assert os.path.exists(plastic_stol)
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
    detdist = abs(DET[0].get_distance())  # approximate for tilted cameras!
    pixsize = DET[0].get_pixel_size()[0]  # assumes square pixel

    theta = np.arcsin(wavelen/2/reso)
    rad = np.tan(2*theta) * detdist/pixsize
    return rad