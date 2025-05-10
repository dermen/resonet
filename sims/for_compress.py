from resonet.sims.simulator import estimate_SNR_per_pixel
from mpi4py import MPI
from resonet.utils.multi_panel import  split_eiger_16M_to_panels
COMM = MPI.COMM_WORLD


def run(args, seeds, jid, njobs):
    """

    :param args: instance of the args() method in this file
    :param jid: job ID
    :param njobs: number of jobs
    :param gvec: randomly rotate the crystals about this axis only
    """
    import sys
    import os
    dirname =os.path.join(os.path.dirname(__file__), "for_tutorial/diffraction_ai_sims_data")
    if not os.path.exists(dirname):
        raise OSError("Please download the simulation data first with the command `resonet-getsimdata`.")
    import time
    import h5py
    import numpy as np
    import dxtbx
    from simtbx.diffBragg import utils
    from scipy.spatial.transform import Rotation
    from scipy.ndimage import binary_dilation

    from resonet.sims.paths_and_const import PDB_MAP
    from resonet.sims import paths_and_const

    from resonet.sims.simulator import Simulator, reso2radius

    IMAX = np.iinfo('uint16').max
    if COMM.rank==0:
        os.makedirs(args.outdir, exist_ok=True)
    COMM.barrier()

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
        from resonet.sims.mosflm_geom import DET, BEAM
        xdim, ydim = DET[0].get_image_size()
        mask = np.ones((ydim, xdim), bool)
    else:
        geom_dirname = os.path.join(os.path.dirname(__file__))
        if args.geom == "pilatus":
            geom_f = os.path.join(geom_dirname, "pilatus_1_00001.cbf")
        elif args.geom == "eiger":
            geom_f = os.path.join(geom_dirname, "eiger_1_00001.cbf")
        else:
            geom_f = os.path.join(geom_dirname, "rayonix_1_00001.cbf")

        if not os.path.exists(geom_f):
            raise OSError(f"Geometry file {geom_f} does not exist, try running `resonet-getsimdata`.")
        loader = dxtbx.load(geom_f)
        DET = loader.get_detector()
        BEAM = loader.get_beam()
        if args.expt is not None:
            from dxtbx.model import ExperimentList
            El = ExperimentList.from_file(args.expt, False)
            DET = El[0].detector
            BEAM = El[0].beam

        # remove the sensor thickness portion of the geometry
        DET = utils.set_detector_thickness(DET)

        raw = loader.get_raw_data().as_numpy_array()
        assert "eiger" in geom_f

        regions, nregions, region_slices, _ = split_eiger_16M_to_panels(raw)
        sY, sX = region_slices[0]
        region_sh = sY.stop-sY.start, sX.stop-sX.start
        assert region_sh == (512, 1028)  # for eiger 16M
        # get the detector dimensions (used to determine detector model below)
        xdim, ydim = DET[0].get_image_size()
        # which pixel do not contain data
        mask = loader.get_raw_data().as_numpy_array() >= 0
        mask = ~binary_dilation(~mask, iterations=2)
        if args.mask is not None:
            mask = np.load(args.mask)
            assert len(mask.shape) == 2

    if jid == 0:
        print("Beginning simulations...")

    # instantiate the simulator class
    HS = Simulator(DET, BEAM, cuda=not args.cpuMode,
                   verbose=args.verbose and jid == 0)
    HS.fix_threefolds = args.fix3fold
    HS.randomize_tilt = args.randTilt
    HS.bg_only = args.bgOnly
    HS.xtal_shape = args.xtalShape
    HS.shots_per_example = args.shotsPerEx
    pixsize = DET[0].get_pixel_size()[0]

    # GPU device Id for this rank
    dev = jid % args.ngpu

    #  how many shots will this rank simulate
    Nshot = len(np.array_split(np.arange(args.nshot), njobs)[jid])

    # write command line info to output folder
    prefix = "compressed"
    if args.noCompress:
        prefix = "rank"
    outname = os.path.join(args.outdir, "%s%d.h5" % (prefix, jid))
    if jid == 0:
        cmd = os.path.join(args.outdir, "commandline.txt")
        config = open(paths_and_const.__file__, 'r').read()
        with open(cmd, "w") as o:
            o.write("working dir: %s\n" % os.getcwd())
            o.write("Python command: " + " ".join(sys.argv) + "\n")
            o.write("\nConfiguration (paths_and_const.py):\n%s" % config)

    with h5py.File(outname, "w") as out:
        out.create_dataset("nominal_mask", data=mask)
        comp_args = {}

        if not args.noCompress:
            comp_args["compression_opts"] = 4
            comp_args["compression"] = "gzip"
            comp_args["shuffle"] = True
        dset_shape = (Nshot*nregions,) + region_sh
        chunks = (1,) + region_sh
        dset = out.create_dataset("images",
                                  shape=dset_shape,
                                  chunks=chunks,
                                  dtype=np.uint16,
                                  **comp_args)

        dset_bg = out.create_dataset("background",
                                  shape=dset_shape,
                                  chunks=chunks,
                                  dtype=np.float16,
                                  **comp_args)

        dset_segments = out.create_dataset("peak_segments",
                                  shape=dset_shape,
                                  chunks=chunks,
                                  dtype=bool,
                                  compression="lzf", shuffle=True)

        param_names = ["reso", "one_over_reso",
                       "radius", "one_over_radius",
                       "is_multi", "multi_lat_angle_sigma",
                       "num_lat", "bg_scale",
                       "beamstop_rad", "detdist", "wavelen",
                       "beam_center_fast", "beam_center_slow",
                       "cent_fast_train", "cent_slow_train",
                       "Na", "Nb", "Nc", "pdb", "mos_spread", "xtal_scale"] \
                      + ["r%d" % x for x in range(1, 10)] + ['pitch_deg', 'yaw_deg', "bg_only"]
        geom_names = ["detdist", "wavelen", "pixsize", "xdim", "ydim", "det_panel", "xstart", "ystart", "i_shot", "rank"]
        lab_dset = out.create_dataset("labels", dtype=np.float32, shape=(Nshot*nregions, len(param_names)), **comp_args)
        geom_dset = out.create_dataset("geom", dtype=np.float32, shape=(Nshot*nregions, len(geom_names)), **comp_args)
        lab_dset.attrs["names"] = param_names
        lab_dset.attrs["pdbmap"] = list(PDB_MAP)
        geom_dset.attrs["names"] = geom_names

        times = []  # store processing times per shot

        # random generators
        random_dist = random_wave = None
        if args.randDist:
            if args.randDistChoice is not None:
                random_dist = lambda: np.random.choice(args.randDistChoice)
            else:
                d1, d2 = args.randDistRange
                assert d1 < d2
                random_dist = lambda: np.random.uniform(d1, d2)
        if args.randWave:
            en1, en2 = args.randWaveRange
            assert en1 < en2
            random_wave = lambda: np.random.uniform(en1, en2)

        rotMats = Rotation.random(Nshot).as_matrix()
        for i_shot in range(Nshot):
            t = time.time()
            pdb_name = args.pdbName
            if pdb_name is not None:
                pdb_name = pdb_name.replace("//", "/")

            # load a mask for this shot
            if maskfiles:
                # choose a random mask for this shot
                maskname = np.random.choice(maskfiles)
                shot_mask = np.load(maskname)
                if jid == 0:
                    print("Loading mask %s" % maskname)
            else:
                shot_mask = mask.copy()

            HS.mask = shot_mask
            if not args.bgOnly and args.randHits:
                HS.bg_only = np.random.choice([0, 1])

            params, spots, imgs, shot_det, shot_beam = HS.simulate(rot_mat=rotMats[i_shot],
                                                                   multi_lattice_chance=args.multiChance,
                                                                   mos_min_max=args.mosMinMax,
                                                                   max_lat=args.maxLat,
                                                                   dev=dev, mos_dom_override=args.nmos,
                                                                   vary_background_scale=args.varyBgScale,
                                                                   pdb_name=pdb_name,
                                                                   randomize_dist=random_dist,
                                                                   randomize_center=args.randCent,
                                                                   randomize_wavelen=random_wave,
                                                                   randomize_scale=args.randScale,
                                                                   low_bg_chance=args.lowBgChance,
                                                                   uniform_reso=args.uniReso,
                                                                   cache_last_img_components=True)

            # at what pixel radius does this resolution corresond to
            radius = reso2radius(params["reso"], DET, BEAM)

            cent_x, cent_y = params["beam_center"]

            # add hot pixels
            npix = imgs[0].size
            if not args.noHot:
                nhot = np.random.randint(0, 6)
                hot_inds = np.random.permutation(npix)[:nhot]

                for i_img, img in enumerate(imgs):
                    img_1d = img.ravel()
                    img_1d[hot_inds] = 2 ** 16
                    img = img_1d.reshape(img.shape)
                    img *= shot_mask
                    imgs[i_img] = img

            # add bad pixels
            if args.noBad:
                min_npix = int(0.01 * xdim)
                max_npix = 3 * min_npix
                nbad = np.random.randint(min_npix, max_npix)
                bad_inds = np.random.permutation(npix)[:nbad]

                for i_img, img in enumerate(imgs):
                    img_1d = img.ravel()
                    img_1d[bad_inds] = 0
                    img = img_1d.reshape(img.shape)
                    img *= shot_mask
                    imgs[i_img] = img

            assert len(imgs)==1
            noise_img = imgs[0]
            snr_pix = estimate_SNR_per_pixel(HS.last_img_spots, HS.last_img_bg, sigma_gain=0.03, sigma_readout=0)
            is_peak = np.logical_and(snr_pix > 0.7 , HS.last_img_spots > 1e-4)
            for i_panel, (sY, sX) in enumerate(region_slices):
                panel_img = noise_img[sY, sX]
                panel_is_peak = is_peak[sY, sX]
                dset_idx = i_shot*nregions + i_panel
                panel_img[panel_img > IMAX] = IMAX
                panel_img = panel_img.astype(np.uint16)
                dset[dset_idx] = panel_img
                panel_img_bg = HS.last_img_bg[sY, sX]
                dset_bg[dset_idx] = panel_img_bg
                dset_segments[dset_idx] = panel_is_peak

            Na, Nb, Nc = params["Ncells_abc"]
            r1, r2, r3, r4, r5, r6, r7, r8, r9 = rotMats[i_shot].ravel()
            if HS.bg_only:
                r1 = r2 = r3 = r4 = r5 = r6 = r7 = r8 = r9 = np.nan
                params["num_lat"] = 0
                params["reso"] = np.nan
                radius = np.nan
                params["multi_lattice"] = 0
                params["ang_sigma"] = np.nan
                Na = Nb = Nc = np.nan
                pdb = np.nan
                params["mos_spread"] = np.nan
                params["crystal_scale"] = np.nan
            else:
                pdb = PDB_MAP[params["pdb_name"]]
            param_arr = [params["reso"], 1 / params["reso"],
                         radius  , 1 / radius,  # TODO update depending on args.centerCrop?
                         params["multi_lattice"],
                         params["ang_sigma"],
                         params["num_lat"],
                         params["bg_scale"],
                         -1,
                         params["detector_distance"],
                         params["wavelength"],
                         cent_x, cent_y,
                         -1, -1,
                         Na, Nb, Nc,
                         pdb,
                         params["mos_spread"],
                         params["crystal_scale"],
                         r1, r2, r3, r4, r5, r6, r7, r8, r9,
                         params['pitch_deg'], params['yaw_deg'],
                         1 if HS.bg_only else 0]

            geom_array = [params["detector_distance"],
                          params["wavelength"],
                          pixsize,
                          xdim, ydim]

            for i_panel, (sY, sX) in enumerate(region_slices):
                xstart = sX.start
                ystart = sY.start
                dset_idx = nregions*i_shot+i_panel
                geom_dset[dset_idx] = geom_array + [i_panel, xstart, ystart, i_shot, COMM.rank]
                lab_dset[dset_idx] = param_arr

            t = time.time() - t
            times.append(t)
            print(
                f"RANK {jid + 1}/{njobs}: Done with shot {i_shot + 1}/{Nshot} out of {args.nshot} total (took {t:.4f} sec).",
                flush=True)

        ave_t = np.mean(times)
        print(
            f"RANK{jid + 1}: Done! Takes {ave_t:.4f} sec on average per image. (Other processes might still be simulating)" % np.mean(
                times))


if __name__=="__main__":
    from resonet.sims.main import args
    import datetime
    timestamp = None
    if COMM.rank==0:
        current_time = datetime.datetime.now()
        timestamp = int(current_time.timestamp())
    timestamp = COMM.bcast(timestamp)
    ap = args()
    if ap.seed is not None:
        timestamp = ap.seed
    seeds = list(range(timestamp,timestamp+COMM.size,1))
    print(f"Rank {COMM.rank} will use seed {seeds[COMM.rank]}")

    run(ap, seeds, COMM.rank, COMM.size)
