
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter as arg_formatter
import sys


def args(use_joblib=False):
    parser = ArgumentParser(formatter_class=arg_formatter)
    parser.add_argument("outdir", help="path to output folder (will be created if necessary)", type=str)
    parser.add_argument("--geom", type=str,
                        help="path to cbf or mccd file for setting the simulation geometry/mask. If None, default geom from mosflm_geom.py will be used.",
                        default=None)
    parser.add_argument("--seed", default=None,
                        help="random number seed. Default value of None will use int(time.time()) . Seed will be offset by MPI rank, so each rank always has a unique seed amongst all ranks.",
                        type=int)
    parser.add_argument("--ngpu", default=1, type=int, help="number of GPUs on machine")
    parser.add_argument("--nshot", default=15000, type=int, help="number of shots to simulate")
    parser.add_argument("--multiChance", type=float, default=0,
                        help="number from 0-1. The probability that a shot will be multi lattice")
    parser.add_argument("--maxLat", type=int, default=3,
                        help="in event of multi lattice shot, this many lattices will be simulated")
    parser.add_argument("--saveRaw", action="store_true", help="Save the raw diffraction images to the hdf5 files")
    parser.add_argument("--mosMinMax", nargs=2, type=float,
                        help="minium and maximum mosaic spread (mosaic spreads wil be drawn randomly, bound by these numbers). Default value of None will use MOS_MIN, MOS_MAX from paths_and_const.py")
    parser.add_argument("--nmos", type=int, default=None,
                        help="Number of mosaic blocks for sampling mosaicity. Default value of None will lead to ~1000 blocks per image (see choose_mos method in make_sims.py).")
    parser.add_argument("--cpuMode", action="store_true",
                        help="run computation on CPU (should specify small --nmos to speed up computation)")
    parser.add_argument("--verbose", action="store_true", help="if true, show extra output (for mpi rank0 only)")
    parser.add_argument("--addHot", action="store_true", help="randomly add hot pixels")
    parser.add_argument("--addBad", action="store_true", help="randomly 0-out bad pixels")
    parser.add_argument("--varyBgScale", action="store_true", help="if true, vary background scale by factor in range 0.05-1.5")
    parser.add_argument("--beamStop", action="store_true", help="if true, add a random beamstop mask to each simulated shot")
    parser.add_argument("--randDist", action="store_true", help="randomize the detector distance")
    parser.add_argument("--randCent", action="store_true", help="randomize the beam center")
    parser.add_argument("--randWave", action="store_true", help="randomize the beam wavelength")
    parser.add_argument("--expt", type=str)
    parser.add_argument("--mask", type=str)
    parser.add_argument("--maskFileList", type=str)
    if use_joblib:
        parser.add_argument("--njobs", default=None, type=int, help="number of jobs")
    args = parser.parse_args()

    if hasattr(args, "h") or hasattr(args, "help"):
        parser.print_help()
        sys.exit()

    return parser.parse_args()


def run(args, seeds, jid, njobs):
    """

    :param args: instance of the args() method in this file
    :param jid: job ID
    :param njobs: number of jobs
    :return:
    """
    import os
    import time
    import h5py
    import numpy as np
    import dxtbx
    from simtbx.diffBragg import utils
    from scipy.spatial.transform import Rotation
    from scipy.ndimage import binary_dilation

    from resonet.sims.simulator import Simulator, reso2radius
    from resonet.utils import eval_model, maxbin
    IMAX = np.iinfo(np.uint8).max

    np.random.seed(seeds[jid])

    maskfiles = []
    if args.maskFileList is not None:
        maskfiles = open(args.maskFileList, "r").readlines()
        maskfiles = [l.strip() for l in maskfiles]
        for m in maskfiles:
            if not os.path.exists(m):
                raise OSError("Not all maskfiles in the maskFileList exist, or the file couldnt be parsed. "
                              "There should be 1 filename per line.")
        if jid==0:
            print("Found %d maskfiles" %len(maskfiles))

    # load the geometry from provided image file
    if args.geom is None:
        from resonet.sims.mosflm_geom import DET,BEAM
        # get the detector dimensions (used to determine detector model below)
        xdim, ydim = DET[0].get_image_size()
        mask = np.ones((ydim, xdim), bool)
    else:
        loader = dxtbx.load(args.geom)
        DET = loader.get_detector()
        BEAM = loader.get_beam()
        if args.expt is not None:
            from dxtbx.model import ExperimentList
            El = ExperimentList.from_file(args.expt, False)
            DET = El[0].detector
            BEAM = El[0].beam

        # remove the sensor thickness portion of the geometry
        DET = utils.set_detector_thickness(DET)

        # get the detector dimensions (used to determine detector model below)
        xdim,ydim = DET[0].get_image_size()
        # which pixel do not contain data
        mask = loader.get_raw_data().as_numpy_array() >= 0
        mask = ~binary_dilation(~mask, iterations=2)
        if args.mask is not None:
            mask = np.load(args.mask)
            assert len(mask.shape) == 2
    factor = 2 if xdim == 2463 else 4
    # make an image whose pixel value corresonds to the radius from the center.
    # and this will be used to create on-the-fly beamstop masks of varying radius
    Y,X = np.indices((ydim, xdim))

    # instantiate the simulator class
    HS = Simulator(DET, BEAM, cuda=not args.cpuMode,
                   verbose=args.verbose and jid==0)

    # sample-to-detector distance and pixel size
    #detdist = abs(DET[0].get_distance())
    pixsize = DET[0].get_pixel_size()[0]
    #wavelen = BEAM.get_wavelength()

    # GPU device Id for this rank
    dev = jid % args.ngpu

    #  how many shots will this rank simulate
    Nshot = len(np.array_split(np.arange(args.nshot), njobs)[jid])

    # write command line info to output folder
    outname = os.path.join(args.outdir, "rank%d.h5" %jid)
    if jid==0:
        cmd = os.path.join(args.outdir, "commandline.txt")
        with open(cmd, "w") as o:
            o.write("working dir: %s\n" % os.getcwd())
            o.write("Python command: " + " ".join(sys.argv) + "\n")

    all_param = []
    all_geom = []
    with h5py.File(outname, "w") as out:
        out.create_dataset("nominal_mask", data=mask)
        dset = out.create_dataset("images",
                                  shape=(Nshot,) + (512,512),
                                  dtype=np.float32)
        # get shape of maximg dset
        test = np.random.random((ydim,xdim))
        test = maxbin.maximg_downsample(test, factor)
        maximg_sh = test.shape
        maximg_dset = out.create_dataset("full_maximg",
                                         shape=(Nshot,) + maximg_sh,
                                         dtype=np.float32)

        if args.saveRaw:
            raw_dset = out.create_dataset("raw_images",
                                          shape=(Nshot,) + (ydim, xdim),
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
                                      dev=dev, mos_dom_override=args.nmos,
                                      vary_background_scale=args.varyBgScale,
                                      randomize_dist=args.randDist,
                                      randomize_center=args.randCent,
                                      randomize_wavelen=args.randWave)

            # at what pixel radius does this resolution corresond to
            radius = reso2radius(params["reso"], DET, BEAM)

            cent_x, cent_y = params["beam_center"]

            # load a mask for this shot
            if maskfiles:
                # choose a random mask for this shot
                maskname = np.random.choice(maskfiles)
                shot_mask = np.load(maskname)
                if jid == 0:
                    print("Loading mask %s" % maskname)
            else:
                shot_mask = mask.copy()
            # add optional beamstop mask:
            beamstop_rad=-1
            if args.beamStop:
                # assume beamstop can vary in radius from 0 to 15 mm
                beamstop_rad_mm = np.random.choice(np.arange(0,15.1,0.375))
                beamstop_rad = int(beamstop_rad_mm/pixsize)

                # jitter the beamstop center by 0.5 mm
                bs_jitt = .5/pixsize
                bs_cent_x = np.random.uniform(cent_x-bs_jitt, cent_x+bs_jitt)
                bs_cent_y = np.random.uniform(cent_y-bs_jitt, cent_y+bs_jitt)
                pixR = np.sqrt((X - bs_cent_x) ** 2 + (Y - bs_cent_y) ** 2)
                is_in_beamstop = pixR < beamstop_rad
                if args.verbose:
                    print("beamstop rad=%.1f" % beamstop_rad)
                shot_mask = np.logical_and(shot_mask, ~is_in_beamstop)

            # add hot pixels
            npix = img.size
            if args.addHot:
                nhot = np.random.randint(0, 6)
                hot_inds = np.random.permutation(npix)[:nhot]

                img_1d = img.ravel()
                img_1d[hot_inds] = 2**16
                img = img_1d.reshape(img.shape)
                img *= shot_mask

            # add bad pixels
            if args.addBad:
                min_npix = int(0.01 * xdim)
                max_npix = 3*min_npix
                nbad = np.random.randint(min_npix, max_npix)
                bad_inds = np.random.permutation(npix)[:nbad]

                img_1d = img.ravel()
                img_1d[bad_inds] = 0
                img = img_1d.reshape(img.shape)
                img *= shot_mask

            # process the raw images according to detector model
            if xdim==2463:  # Pilatus 6M
                quad = eval_model.raw_img_to_tens_pil(img, shot_mask, xy=(int(round(cent_x)), int(round(cent_y))))
            elif xdim==4096:  # Mar
                quad = eval_model.raw_img_to_tens_mar(img, shot_mask)
            else:  # Eiger
                quad = eval_model.raw_img_to_tens(img, shot_mask)

            # convert cent_x, cent_y to downsampled version
            cent_x_train = (cent_x - xdim*.5)/factor
            cent_y_train = (cent_y - ydim*.5)/factor
            all_param.append(
                [params["reso"], 1/params["reso"],
                 radius/factor, factor/radius,
                 params["multi_lattice"],
                 params["ang_sigma"],
                 params["num_lat"],
                 params["bg_scale"],
                 beamstop_rad,
                 params["detector_distance"],
                 params["wavelength"],
                 cent_x, cent_y,
                 cent_x_train, cent_y_train])
            param_names = ["reso", "one_over_reso",
                           "radius", "one_over_radius",
                           "is_multi", "multi_lat_angle_sigma",
                           "num_lat", "bg_scale",
                           "beamstop_rad", "detdist", "wavelen",
                           "beam_center_fast", "beam_center_slow",
                           "cent_fast_train", "cent_slow_train"]
            all_geom.append([params["detector_distance"],
                             params["wavelength"],
                             pixsize,
                             xdim, ydim])

            if args.saveRaw:
                raw_dset[i_shot] = img
            maximg = maxbin.maximg_downsample(img*shot_mask, factor)
            # normalize and convert
            maximg[ maximg < 0] = 0
            maximg = np.sqrt(maximg)
            maximg[maximg > IMAX] = IMAX
            maximg = maximg.astype(np.float32)
            maximg_dset[i_shot] = maximg

            dset[i_shot] = quad
            t = time.time()-t
            times.append(t)
            if jid == 0:
                print("Done with shot %d / %d (took %.4f sec)" % (i_shot+1, Nshot, t), flush=True)
        lab_dset = out.create_dataset("labels", data=all_param)
        lab_dset.attrs["names"] = param_names
        geom_dset = out.create_dataset("geom", data=all_geom, dtype=np.float32)
        geom_dset.attrs["names"] = ["detdist", "wavelen", "pixsize", "xdim", "ydim"]
        if jid == 0:
            print("Done! Takes %.4f sec on average per image" % np.mean(times))
