
import os
import time
from copy import deepcopy
from simtbx.diffBragg.utils import ENERGY_CONV
from dxtbx.model import Detector, Panel
import numpy as np
from scipy.spatial.transform import Rotation

from scitbx.array_family import flex
from simtbx.nanoBragg import nanoBragg, sim_data
from simtbx import get_exascale
from simtbx.nanoBragg.utils import get_xray_beams


from resonet.sims import make_sims, make_crystal, paths_and_const


class Simulator:

    def __init__(self, DET, BEAM, cuda=True, verbose=False):
        """
        :DET: dxtbx detector model
        :BEAM: dxtbx beam model
        :cuda: use cuda backend for simulation of spots (and backgrond if the exafel api is avail)
        :verbose:
        """
        self.DET = DET
        self.BEAM = BEAM
        # air and water background:
        self.air_and_water = make_sims.get_background(self.DET, self.BEAM)
        # stol of every pixel:
        self.STOL = make_sims.get_theta_map(self.DET, self.BEAM)

        self.nb_beam = make_crystal.load_beam(self.BEAM,
                                      divergence=paths_and_const.DIVERGENCE_MRAD / 1e3 * 180 / np.pi)
        self.cuda=cuda
        self.verbose = verbose
        self.shot_det = self.shot_beam = None   # dxtbx detector and beam of latest shot
        self.bg_only = False  # only simulate background
        self.randomize_tilt = False  # random tilt angle per shot
        self.fix_threefolds = False  # correct F_latt for trigonal / hexagonal space groups (doesnt matter if using gauss_star shape)
        self.xtal_shape = "gauss"  # shape of the RELP, can be square, gauss, or gauss_star
        self.shots_per_example = 1  # if provided simulate will return multiple images, each with a different crystal orientation and noise
        self.mask = None  # place holder for numpy-style pixel mask
        self.gpud = self.exascale_api = None
        if self.cuda:
            try:
                self.gpud = get_exascale("gpu_detector", "cuda")
                self.exascale_api = get_exascale("exascale_api", "cuda")
            except ImportError:
                print("Warning, simtbx_gpu_ext not installed, background simulation will be slow!")

    def simulate(self, rot_mat=None, multi_lattice_chance=0, max_lat=2, mos_min_max=None,
                 pdb_name=None, plastic_stol=None, dev=0, mos_dom_override=None, vary_background_scale=False,
                 randomize_dist=None, randomize_wavelen=None, randomize_center=False,
                 randomize_scale=False, low_bg_chance=0, uniform_reso=False, roi=None,
                 old_multi_spread=True, cbf_name=None):
        """

        :param rot_mat: specific orientation matrix for crystal
        :param multi_lattice_chance:  probabilitt to include more lattices
        :param max_lat: maximum number of lattices per shot (will be chosen randomly if > 1)
        :param mos_min_max: values for determing the size of the mosaic spread. Should be lower, upper bounds given in degrees 
        :param pdb_name: path to the pdb subfolder (for debug purposes).
        :param plastic_stol: path to the plastic `sin theta over lambda` file (debug purposes only)
        :param dev: device id (number from 0 to N-1 where N is number of nvidia GPUs available (run nvidia-smi to check)
        :param mos_dom_override: number of mosaic blocks to simulate. If not, will be determined via make_sims.choose_mos function.
        :param vary_background_scale: whether to vary the scale of background from shot-to-shot
        :param randomize_dist: a function that returns a randomized detector distance
        :param randomize_wavelen: a function that returns a randomized beam energy
        :param randomize_center: randomize the beam center
        :param randomize_scale: randomize the crystal domain size
        :param low_bg_chance: probability to simulate a low backgroun shot
        :param uniform_reso: sample resolution uniformly to edge of camera
        :roi: a region of interest to simulate (experimental)
        :old_multi_spread: use the original routine for generating random angles between overlapping lattices
           This was the method used for the multi lattice model reported on in: https://doi.org/10.1107/S2059798323010586 
        :cbf_name: if provided, a raw CBF image will be written. It can be read with ADXV and dials.image_viewer
            but not the python package fabio (for reasons unknown)
        :return: parameters and simulated image
        """
        if multi_lattice_chance > 0:
            assert self.shots_per_example == 1
        if multi_lattice_chance > 0 or self.shots_per_example > 1:
            assert not paths_and_const.LAUE_MODE
            assert not self.fix_threefolds
        if pdb_name is None:
            pdb_name = make_sims.choose_pdb()
        else:
            assert os.path.exists(pdb_name)
            assert os.path.isdir(pdb_name)
        crystal_scale = 1
        if randomize_scale:
            crystal_scale = np.random.choice([1,1,1,1,2,2,3])
        C = make_crystal.load_crystal(pdb_name, rot_mat, crystal_scale, cut_1p2=paths_and_const.CUT_1P2,
                                      xtal_shape=self.xtal_shape)
        mos_min = mos_max = None  # will default to values in paths_and_const.py
        if mos_min_max is not None:
            mos_min, mos_max = mos_min_max

        mos_spread, mos_dom = make_sims.choose_mos(mos_min, mos_max)
        C.mos_spread_deg = mos_spread

        if mos_dom_override is not None:
            mos_dom = mos_dom_override
        if mos_dom != 1:
            n_mos = mos_dom//2
            if n_mos % 2 == 1:
                n_mos += 1
        else:
            n_mos = 1
        C.n_mos_domains = n_mos
        if n_mos==1:
            C.mos_spread_deg = 0

        S = sim_data.SimData()
        S.crystal = C

        pitch_angle = yaw_angle = 0

        if randomize_dist is not None or randomize_center or randomize_wavelen is not None or self.randomize_tilt:
            shot_beam = deepcopy(self.BEAM)
            if randomize_wavelen is not None:
                energy_ev = randomize_wavelen()
                shot_wavelen = ENERGY_CONV / energy_ev
                shot_beam.set_wavelength(shot_wavelen)

            shot_det = deepcopy(self.DET)
            if randomize_dist is not None:
                curr_dist = self.DET[0].get_distance()
                new_dist = randomize_dist()
                dist_shift = new_dist - curr_dist
                shot_det = shift_distance(shot_det, dist_shift)

            if randomize_center:
                cent_window_mm = paths_and_const.CENTER_WINDOW_MM
                cent_window_pix = int(cent_window_mm/shot_det[0].get_pixel_size()[0])
                new_cent_x, new_cent_y = np.random.uniform(-cent_window_pix, cent_window_pix, 2)
                shot_det = shift_center(shot_det, new_cent_x, new_cent_y)

            if self.randomize_tilt:
                pitch = paths_and_const.RANDOM_TILT_PITCH_DEG*np.pi/180
                yaw = paths_and_const.RANDOM_TILT_YAW_DEG *np.pi/180
                pitch_angle = np.random.uniform(-pitch, pitch)
                yaw_angle = np.random.uniform(-yaw, yaw)
                fast_axis = shot_det[0].get_fast_axis()
                slow_axis = shot_det[0].get_slow_axis()
                shot_det.rotate_around_origin(fast_axis,pitch_angle)
                shot_det.rotate_around_origin(slow_axis,yaw_angle)

            if self.gpud is None:
                shot_air_and_water = make_sims.get_background(shot_det, shot_beam, roi=roi)
            redo_air_water = True

            # stol of every pixel:
            STOL = make_sims.get_theta_map(shot_det, shot_beam)

            nb_beam = make_crystal.load_beam(shot_beam,
                                          divergence=paths_and_const.DIVERGENCE_MRAD / 1e3 * 180 / np.pi)

        else:
            shot_air_and_water = None
            redo_air_water = False
            air_and_water = self.air_and_water
            STOL = self.STOL
            nb_beam = self.nb_beam
            shot_beam = self.BEAM
            shot_det = self.DET

        S.beam = nb_beam
        S.detector = shot_det
        S.instantiate_nanoBragg(oversample=1)

        # for this detector, this is the minimum resolution allowed
        high_reso = shot_det.get_max_resolution(shot_beam.get_s0())
        if roi is not None:
            S.D.region_of_interest = roi 
        if self.verbose:
            S.D.show_params()
            print("Simulating spots!", flush=True)
        S.D.device_Id = dev
        if self.fix_threefolds:
            num_blocks = len(S.D.get_mosaic_blocks())
            p1_cryst = deepcopy(C.dxtbx_crystal)
            ref_cryst = p1_cryst.change_basis(C.space_group_info.change_of_basis_op_to_primitive_setting().inverse())
            S.D.set_mosaic_blocks_sym(ref_cryst, reference_symbol=C.symbol, orig_mos_domains=num_blocks)
        if self.cuda and S.D.add_nanoBragg_spots_cuda is None:
            print("Warning: Trying to use CUDA, but no simtbx CUDA install available.")
            self.cuda = False

        if self.cuda:
            S.D.device_Id = dev
            S.D.add_nanoBragg_spots_cuda()
        else:
            S.D.add_nanoBragg_spots()
        spots = S.D.raw_pixels.as_numpy_array()
        xdim, ydim = shot_det[0].get_image_size()
        img_sh = ydim, xdim
        spots = spots.reshape(img_sh)
        use_multi = np.random.random() < multi_lattice_chance
        ang_sigma = 0
        num_additional_lat = 0
        extra_shots = []
        if use_multi or self.shots_per_example > 1:
            nominal_crystal = S.crystal.dxtbx_crystal
            Umat = np.reshape(nominal_crystal.get_U(), (3, 3))
            if use_multi:
                num_additional_lat = np.random.choice(range(1,max_lat))
                mats = Rotation.random(num_additional_lat).as_matrix()
                vecs = np.dot(np.array([1, 0, 0])[None], mats)[0]

                if old_multi_spread:
                    ang_sigma = np.random.choice([0.1,1,10])
                    angs = np.random.normal(0, ang_sigma, num_additional_lat)
                else:
                    angs = np.random.uniform(1, 180, num_additional_lat)
                    ang_sigma = np.std(np.append(angs,[0]))
                scaled_vecs = vecs*angs[:, None]
                rot_mats = Rotation.from_rotvec(scaled_vecs).as_matrix()
            else:
                num_additional_lat = self.shots_per_example - 1
                rot_mats = Rotation.random(num_additional_lat).as_matrix()

            for i_p, perturb in enumerate(rot_mats):
                if self.verbose:
                    print("additional multi lattice sim %d" % (i_p+1), flush=True)
                Umat_p = np.dot(perturb, Umat)
                nominal_crystal.set_U(tuple(Umat_p.ravel()))
                S.D.Amatrix = sim_data.Amatrix_dials2nanoBragg(nominal_crystal)

                S.D.raw_pixels *= 0
                if self.cuda:
                    S.D.add_nanoBragg_spots_cuda()
                else:
                    S.D.add_nanoBragg_spots()

                this_latt_spots = S.D.raw_pixels.as_numpy_array()
                if use_multi:
                    spots += this_latt_spots
                else:
                    extra_shots.append(this_latt_spots)
            # only normalize if we summed the shots (multi-lattice mode)
            if use_multi:
                spots /= (num_additional_lat+1)

        all_spots = [spots] + extra_shots

        if self.verbose:
            print("sim random bg", flush=True)
        if plastic_stol is None:
            plastic_stol = np.random.choice(paths_and_const.RANDOM_STOLS)
        else:
            assert os.path.exists(plastic_stol)

        if uniform_reso:
            reso, Bfac_img = make_sims.get_Bfac_img(STOL,high_reso)
        else:
            reso, Bfac_img = make_sims.get_Bfac_img(STOL)

        if self.gpud is None:
            plastic = make_sims.random_bg(shot_det, shot_beam, plastic_stol, roi=roi)
            bg = plastic + shot_air_and_water
        else:
            bg = self.sim_background(shot_det, shot_beam, dev, plastic_stol, redo_air_water) 

        if paths_and_const.FLAT_BACKGROUND:
            bg = np.ones_like(bg)* np.mean(bg)
        bg_scale = 1
        if vary_background_scale:
            
            low_bg_scale = np.random.choice([0.01, 0.05, 0.05,  0.1])
            norm_bg_scale = np.random.choice([1, 1, 1.25, 1.5, 2])
            is_low_bg = np.random.random() < low_bg_chance
            bg_scale = low_bg_scale if is_low_bg else norm_bg_scale
            if self.verbose:
                print("Scaling background by %.3f" % bg_scale)

        make_sims.set_noise(S.D)
        noise_imgs = []
        all_spots_scaled = []
        for spots in all_spots:
            spots_scaled = Bfac_img*paths_and_const.VOL*spots
            all_spots_scaled.append(spots_scaled)
            if self.bg_only:
                img = bg*bg_scale
            else:
                img = spots_scaled + bg*bg_scale

            S.D.raw_pixels = flex.double(img.ravel())
            S.D.add_noise()
            noise_img = S.D.raw_pixels.as_numpy_array().reshape(img_sh)
            noise_imgs.append(noise_img)

        param_dict = {"reso": reso,
                      "multi_lattice": use_multi,
                      "ang_sigma": ang_sigma,
                      "bg_scale": bg_scale,
                      "num_lat": num_additional_lat+1,
                      "wavelength": nb_beam.spectrum[0][0],
                      "detector_distance": S.detector[0].get_distance(),
                      "beam_center": S.detector[0].get_beam_centre_px(nb_beam.unit_s0),
                      "Ncells_abc": C.Ncells_abc,
                      "pdb_name": pdb_name, 
                      "mos_spread": mos_spread,
                      "crystal_scale": crystal_scale,
                      "Umat": S.crystal.dxtbx_crystal.get_U(),
                      "pitch_deg": pitch_angle*180/np.pi,
                      "yaw_deg": yaw_angle*180/np.pi,
                      "wavelen_data": None}

        if cbf_name:
            raw_pix = deepcopy(S.D.raw_pixels)
            if self.mask is not None:
                raw_pix = raw_pix.as_numpy_array().ravel()
                raw_pix[~self.mask.ravel()] = -1
                raw_pix = flex.double(raw_pix)
            raw_pix.resize(flex.grid((ydim, xdim)))
            S.D.raw_pixels = raw_pix
            S.D.to_cbf(cbf_name, toggle_conventions=True)
        S.D.free_all()

        self.shot_det = shot_det
        self.shot_beam = shot_beam
        return param_dict, all_spots_scaled, noise_imgs, shot_det, shot_beam

    def sim_background(self, det, beam,
                dev, stol_name, redo_air_water=False):
        """
        det: dxtbx detector
        beam: dxtbx beam
        dev: gpu device Id
        stol_name: string path of a sin-theta-over-lambda file
        redo_air_water: redo the air and water simulation (assuming det/beam changed)
        returns np.ndarray of background scattering pixels
        """

        total_flux = paths_and_const.FLUX
        spectrum = [(beam.get_wavelength(), 1)]
        xray_beams = get_xray_beams(spectrum, beam)

        SIM = nanoBragg(det, beam, panel_id=0)
        SIM.beamsize_mm = paths_and_const.BEAM_SIZE_MM
        SIM.xray_beams = xray_beams
        SIM.flux = paths_and_const.FLUX
        SIM.device_Id = dev

        SIM.Fbg_vs_stol = make_sims.load_stol(stol_name)
        SIM.amorphous_sample_thick_mm = paths_and_const.XTALSIZE_MM
        SIM.amorphous_density_gcm3 = 1
        SIM.amorphous_molecular_weight_Da = 12

        gpu_simulation = self.exascale_api(nanoBragg=SIM)
        gpu_simulation.allocate()
        gpu_detector = self.gpud(deviceId=dev, detector=det, beam=beam)
        gpu_detector.each_image_allocate()
        #gpu_detector.setup_random_states()  # how long is this
        #zeros = flex.double(np.zeros(spots_scaled.size))
        #gpu_detector.set_raw_pixels(zeros)
        gpu_detector.scale_in_place(0)

        # add the plastic
        gpu_simulation.add_background(gpu_detector)

        if redo_air_water:
            # AIR
            SIM.Fbg_vs_stol = make_sims.load_stol(paths_and_const.AIR_STOL)
            SIM.amorphous_sample_thick_mm = 5
            SIM.amorphous_density_gcm3 = 1.2e-3
            SIM.amorphous_molecular_weight_Da = 14  # nitrogen = N2
            gpu_simulation.add_background(gpu_detector)

            # WATER
            SIM.Fbg_vs_stol = make_sims.load_stol(paths_and_const.WATER_STOL)
            SIM.amorphous_sample_thick_mm = paths_and_const.XTALSIZE_MM
            SIM.amorphous_density_gcm3 = 1
            SIM.amorphous_molecular_weight_Da = 18
            gpu_simulation.add_background(gpu_detector)
            water = gpu_detector.get_raw_pixels().as_numpy_array()

        #flex_spots = flex.double(spots_scaled)
        #gpu_detector.offset_in_place(flex_spots)

        # NOISE:
        #SIM.detector_calibration_noise_pct = 3
        #SIM.adc_offset_adu = 0
        #SIM.quantum_gain = 1
        #SIM.readout_noise_adu = 0
        #gpu_detector.noisify(SIM.flicker_noise_pct, SIM.detector_calibration_noise_pct/100., SIM.readout_noise_adu,
        #                     SIM.quantum_gain, SIM.adc_offset_adu, 0)
        nominal_data = gpu_detector.get_raw_pixels().as_numpy_array()
        gpu_detector.each_image_free()  # deallocate GPU arrays
        #gpu_detector.free_random_states()
        return nominal_data


def reso2radius(reso, DET, BEAM):
    """

    :param reso: resolution of a position on camera in Angstrom
    :param DET: dxtbx detector model
    :param BEAM: dxtbx beam model
    :return: the resolution converted to pixel radii (distance from beam center)
    """
    wavelen = BEAM.get_wavelength()
    detdist = abs(DET[0].get_distance())
    pixsize = DET[0].get_pixel_size()[0]  # assumes square pixel

    theta = np.arcsin(wavelen/2/reso)
    rad = np.tan(2*theta) * detdist/pixsize
    return rad


def shift_center(det, delta_x, delta_y):
    """
    :param det: dxtbx detector model (single panel)
    :param delta_x: beam center shift in pixels (fast dim)
    :param delta_y: beam center shift in pixels (slow dim)
    :return: dxtbx detector model with shifted center
    """
    dd = det[0].to_dict()
    F = np.array(dd["fast_axis"])
    S = np.array(dd["slow_axis"])
    O = np.array(dd['origin'])
    pixsize = det[0].get_pixel_size()[0]
    O2 = O + F * pixsize * delta_x + S * pixsize * delta_y
    dd["origin"] = tuple(O2)
    new_det = Detector()
    new_pan = Panel.from_dict(dd)
    new_det.add_panel(new_pan)
    return new_det

def shift_distance(det, delta_z):
    """
    :param det: dxtbx detector model (single panel)
    :param delta_z: distance shift in millimeters
    :return: dxtbx detector model with shifted center
    """
    dd = det[0].to_dict()
    F = np.array(dd["fast_axis"])
    S = np.array(dd["slow_axis"])
    O = np.array(dd['origin'])
    Orth = np.cross(F,S)
    O2 = O + Orth*delta_z
    dd["origin"] = tuple(O2)
    new_det = Detector()
    new_pan = Panel.from_dict(dd)
    new_det.add_panel(new_pan)
    return new_det
