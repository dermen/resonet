# coding: utf-8
import glob
import numpy as np
from scipy.interpolate import interp1d
import time

import dxtbx
from scitbx.array_family import flex
from simtbx.diffBragg import utils
from simtbx.nanoBragg import utils as nb_utils
from simtbx.nanoBragg import sim_data


from resonet.sims import paths_and_const
from resonet.sims import make_crystal


def choose_res(hres=None):
    """ choose a random resolution"""
    if paths_and_const.FIX_RES is not None:
        return paths_and_const.FIX_RES
    if hres is None:
        res = 0.5/(paths_and_const.STOL_MIN + np.random.random()*paths_and_const.STOL_RNG)
    else:
        res = np.random.uniform(hres, 6)  # maximum of 6 Angstrom  
    return res


def choose_pdb():
    # choose a random PDB file
    return np.random.choice(paths_and_const.RANDOM_PDBS)


def choose_deltaB(hres=None):
    # choose a delta -Bfactor to scale the scattering
    res = choose_res(hres)
    B = 4*res**2 + 12
    return B-10


def choose_mos(mos_min=None, mos_max=None):
    # choose a random mosaicity
    if mos_min is None:
        mos_min = paths_and_const.MOS_MIN
    if mos_max is None:
        mos_max = paths_and_const.MOS_MAX
    assert mos_min < mos_max

    mos_rng = (np.sqrt(mos_max) - np.sqrt(mos_min))**2    
    r = np.random.random()
    mosaic = mos_min + r*mos_rng
    mosdoms = int(1000 + 50*mosaic**2)
    return mosaic, mosdoms


def choose_stol():
    # choose a random plastic scattering profile
    stol, Fbg = np.loadtxt(np.random.choice(paths_and_const.RANDOM_STOLS)).T
    flex.vec2_double(list(zip(Fbg, stol)))
    return flex.vec2_double( Fbg, stol)


def random_bg(D,B, stol_name, roi=None):
    """

    :param D: dxtbx detector
    :param B: dxtbx beam
    :param stol_name: flex.vec2d scattering profile (sin theta vs lambda)
    :param  roi: nanoBragg region_of_interest
    :return:
    """
    # simulate scattering from a plastic scattering profile
    funky_bg = nb_utils.sim_background(D, B, [B.get_wavelength()], [1], paths_and_const.FLUX,
                molecular_weight=12, sample_thick_mm=paths_and_const.XTALSIZE_MM,
                Fbg_vs_stol=load_stol(stol_name), roi=roi)
    return funky_bg.as_numpy_array()


def load_stol(name):
    """

    :param name:  name of a stol text file (first col is sqrt(intensity), second is the sin-theta-over-lambda value)
    :return:
    """
    Fbg, stol = np.loadtxt(name).T
    return flex.vec2_double(list(zip(Fbg, stol)))


def get_background(D,B, no_air=False, no_water=False, water_path_mm=None, air_path_mm=None, roi=None):

    air = 0
    if not no_air:
        if air_path_mm is None:
            air_path_mm = 5
        air = nb_utils.sim_background(D, B, [B.get_wavelength()], [1], paths_and_const.FLUX, molecular_weight=14,
                                      sample_thick_mm=air_path_mm,
                                   Fbg_vs_stol=load_stol(paths_and_const.AIR_STOL), density_gcm3=1.2e-3, roi=roi)  #

    water = 0
    if not no_water:
        if water_path_mm is None:
            water_path_mm = paths_and_const.XTALSIZE_MM
        water = nb_utils.sim_background(D, B, [B.get_wavelength()], [1], paths_and_const.FLUX, molecular_weight=18,
                                   sample_thick_mm=water_path_mm,
                                   Fbg_vs_stol=load_stol(paths_and_const.WATER_STOL), density_gcm3=1, roi=roi)  #

    background = air + water
    return background.as_numpy_array()


def get_Bfac_img(STOL, hres=None):
    """

    :param STOL: sin-theta-over-lambda of every pixel on detector
    :return: delta-Bfactor at every pixel (for aadjusting the spot resolution)
    """
    B, stol, factor = get_deltaB_factor(hres)
    I = interp1d(stol, factor, bounds_error=False, fill_value=0)
    Bfac_img = I(STOL.ravel()).reshape(STOL.shape)
    reso = np.sqrt(.25*(B + 10 - 12))
    return reso, Bfac_img


def get_deltaB_factor(hres=None):
    """

    :return: 3-tuple,
        -first element is the delta-B factor (can be converted to resolution)
        -second element is the sin-theta-over-lambda values
        -third element is the B-factor scale at each sin-theta-over-lambda value
    """
    stol = np.arange(0, 0.5, 0.01)
    B = choose_deltaB(hres)
    exponent = 2 * B * stol ** 2
    is_bad = exponent > 100
    fac = np.exp(-exponent)
    fac[is_bad] = 0
    return B, stol, fac


def get_theta_map(detector, beam):
    """

    :param detector: dxtbx detector
    :param beam: dxtbx beam
    :return: sin-theta-over-lambda for each pixel
    """
    Qmags = {}
    DIFFRACTED = {}
    AIRPATH ={}
    unit_s0 = beam.get_unit_s0()
    for pid in range(len(detector)):
        xdim, ydim = detector[pid].get_image_size()
        panel_sh = ydim, xdim

        FAST = np.array(detector[pid].get_fast_axis())
        SLOW = np.array(detector[pid].get_slow_axis())
        ORIG = np.array(detector[pid].get_origin())

        Ypos, Xpos = np.indices(panel_sh)
        px = detector[pid].get_pixel_size()[0]
        Ypos = Ypos* px
        Xpos = Xpos*px

        SX = ORIG[0] + FAST[0]*Xpos + SLOW[0]*Ypos
        SY = ORIG[1] + FAST[1]*Xpos + SLOW[1]*Ypos
        SZ = ORIG[2] + FAST[2]*Xpos + SLOW[2]*Ypos
        AIRPATH[pid] = np.sqrt(SX**2 + SY**2 + SZ**2)   # units of mm

        Snorm = np.sqrt(SX**2 + SY**2 + SZ**2)

        SX /= Snorm
        SY /= Snorm
        SZ /= Snorm

        DIFFRACTED[pid] = np.array([SX, SY, SZ])

        QX = (SX - unit_s0[0]) / beam.get_wavelength()
        QY = (SY - unit_s0[1]) / beam.get_wavelength()
        QZ = (SZ - unit_s0[2]) / beam.get_wavelength()
        Qmags[pid] = np.sqrt(QX**2 + QY**2 + QZ**2)

    Qmags = Qmags[0]  # only working with single panel dets
    STOL = Qmags/2
    return STOL


def set_noise(noise_sim, calib_noise_percent=3):
    """

    :param noise_sim: nanoBragg simulator instance
    :param calib_noise_percent: calibration noise (how much each pixels gain varies)
    :return: nanoBragg simulator instance
    """
    #noise_sim = nanoBragg(detector=DET, beam=BEAM)
    #noise_sim.beamsize_mm = paths_and_const.BEAM_SIZE_MM
    noise_sim.detector_calibration_noise_pct = calib_noise_percent
    noise_sim.exposure_s = 1
    noise_sim.calib_seed=0
    noise_sim.seed=0
    #noise_sim.flux = paths_and_const.FLUX
    noise_sim.adc_offset_adu =0
    noise_sim.detector_psf_kernel_radius_pixels = 5
    noise_sim.detector_psf_fwhm_mm =0
    noise_sim.quantum_gain = 1
    noise_sim.readout_noise_adu = 0
    return noise_sim


def main():
    fnames = glob.glob("/mnt/data/s2/blstaff/SOLTIS/AI_PREDICTION/3.15A/*cbf")
    loader = dxtbx.load(fnames[0])
    D = loader.get_detector()
    D0 = utils.set_detector_thickness(D)

    C = make_crystal.load_crystal("pdbs/3t4x")
    C.mos_spread_deg = 2.60736
    C.n_mos_domains = 1340//2

    S = sim_data.SimData()
    S.crystal = C
    S.beam= make_crystal.load_beam(loader.get_beam())
    S.detector = D0
    S.instantiate_nanoBragg(oversample=1)

    S.D.divergence_hv_mrad = 2e-5,2e-5
    S.D.divsteps_hv = 1,1
    S.D.show_params()

    S.D.add_nanoBragg_spots_cuda()
    img = S.D.raw_pixels.as_numpy_array()
    vol = 125000000
    img *= vol
    print("done")

    #import pylab as plt
    #plt.imshow(img * 125000000, vmax=100000)
    #plt.show()

if __name__=="__main__":
    main()
