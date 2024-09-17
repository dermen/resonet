


def args(use_joblib=False):
    from argparse import ArgumentParser
    from argparse import ArgumentDefaultsHelpFormatter as arg_formatter
    parser = ArgumentParser(formatter_class=arg_formatter)
    parser.add_argument("outdir", help="path to output folder (will be created if necessary)", type=str)
    parser.add_argument("--geom", type=str, choices=["eiger", "pilatus", "mar"], help="available detector formats (`eiger`, `pilatus`, or `mar`)", default=None)
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
    parser.add_argument("--noHot", action="store_true", help="dont randomly add hot pixels")
    parser.add_argument("--noBad", action="store_true", help="dont randomly 0-out pixels")
    parser.add_argument("--varyBgScale", action="store_true", help="if true, vary background scale by factor in range 0.05-1.5")
    parser.add_argument("--beamStop", action="store_true", help="if true, add a random beamstop mask to each simulated shot")
    parser.add_argument("--randDist", action="store_true", help="randomize the detector distance")
    parser.add_argument("--randDistRange", nargs=2, type=float, default=[200,300], help="If --randDist, and if --randDistChoice is not provided, then detdist will be drawn uniformly in this range for each shot")
    parser.add_argument("--randDistChoice", default=None, nargs="+", type=float, help="If provided, and if --randDist, then detdist will be chosen randomly for each shot from these values (default is None)")
    parser.add_argument("--randWaveRange", nargs=2, type=float, default=[10000,13000], help="if randWave, then energies will be drawn uniformly in this range for each shots wavelength")
    parser.add_argument("--randCent", action="store_true", help="randomize the beam center")
    parser.add_argument("--randAxis", action="store_true", help="a random axis will be chosen, then, for all simulations, each crystal will be rotated a random amount about that axis. This supercedes twoAxisOnly and axisRotOnly")
    parser.add_argument("--randWave", action="store_true", help="randomize the beam wavelength")
    parser.add_argument("--randScale", action="store_true", help="randomize the crystal domain size")
    parser.add_argument("--axisRotOnly", choices=[0,1,2], type=int, default=None, help="Rotate the crystals about soecified axis (as a control)")
    parser.add_argument("--twoAxisOnly", choices=[0,1,2], type=int, default=None, help="Rotate the crystals about specified axes (as a control) (0=xy, 1=xz, 2=yz)")
    parser.add_argument("--expt", type=str)
    parser.add_argument("--mask", type=str)
    parser.add_argument("--maskFileList", type=str)
    parser.add_argument("--pdbName", type=str, default=None, help="if provided all simulations will use crystal model from this PDB")
    parser.add_argument("--lowBgChance", type=float, default=0, help="probability to simulate a log background shot (default=0)")
    parser.add_argument("--uniReso", action="store_true", help="uniformly sample resolution per shot, up to the detector maximum")
    parser.add_argument("--noCompress", action="store_true", help="store uncompressed files")
    parser.add_argument("--centerCrop", action="store_true", help="Alternative to quad downsampling, downsample whole image by a factor and "
                                                                  "crop around the center")
    parser.add_argument("--sanityTestOps", action="store_true", help="If True, then ensure application of operators in the SGOPS file produce the same diffraction pattern")
    parser.add_argument("--bgOnly", action="store_true", help="Only simulate background scattering")
    parser.add_argument("--randTilt", action="store_true", help="Randomize detector tilt angles (angles stored as pitch/yaw labels)")
    parser.add_argument("--fix3fold", action="store_true", help="Ensure F_latt is 3-fold symmetric (experimental)")
    parser.add_argument("--xtalShape", default="gauss", type = str, help="shape factor of the relp, can be gauss, square, or gauss_star (default=gauss)")
    parser.add_argument("--shotsPerEx", default=1, type=int, help="number of shots per example, if more than 1, each shot will have same params but a random Umat")
    parser.add_argument("--randHits", action="store_true", help="generate diffraction+background images and background-only images with equal probability")
    if use_joblib:
        parser.add_argument("--njobs", default=None, type=int, help="number of jobs")
    args = parser.parse_args()

    if hasattr(args, "h") or hasattr(args, "help"):
        parser.print_help()
        sys.exit()

    return parser.parse_args()


def run(args, seeds, jid, njobs, gvec=None):
    """

    :param args: instance of the args() method in this file
    :param jid: job ID
    :param njobs: number of jobs
    :param gvec: randomly rotate the crystals about this axis only
    """
    import sys
    import os
    dirname=os.path.join(os.path.dirname(__file__), "for_tutorial/diffraction_ai_sims_data")
    if not os.path.exists(dirname):
        raise OSError("Please download the simulation data first with the command `resonet-getsimdata`.")
    import time
    import h5py
    import numpy as np
    import dxtbx
    from simtbx.diffBragg import utils
    from scipy.spatial.transform import Rotation
    from scipy.ndimage import binary_dilation
    import torch
    
    from resonet.sims.paths_and_const import PDB_MAP
    from resonet.utils.eval_model import to_tens
    from resonet.utils import counter_utils
    from resonet.sims import paths_and_const

    from resonet.sims.simulator import Simulator, reso2radius

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
        xdim, ydim = DET[0].get_image_size()
        mask = np.ones((ydim, xdim), bool)
    else:
        geom_dirname=os.path.join(os.path.dirname(__file__))
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

        # get the detector dimensions (used to determine detector model below)
        xdim,ydim = DET[0].get_image_size()
        # which pixel do not contain data
        mask = loader.get_raw_data().as_numpy_array() >= 0
        mask = ~binary_dilation(~mask, iterations=2)
        if args.mask is not None:
            mask = np.load(args.mask)
            assert len(mask.shape) == 2

    geom_dict = {"detector":DET, "beam":BEAM}

    # TODO: check whether factor is meant to be replaced totally by quad_ds_fact, and adjust rest of code accordingly
    # process the raw images according to detector model
    if xdim == 2463:  # Pilatus 6M
        quad_ds_fact = 2
        center_ds_fact = 3
    elif xdim == 3840:
        quad_ds_fact = 3
        center_ds_fact = 4
    elif xdim == 4096:  # Mar
        quad_ds_fact = 4
        center_ds_fact = 5
    else:  # Eiger
        quad_ds_fact = 4
        center_ds_fact = 5
    cropdim = min(xdim, ydim) // center_ds_fact - 1
    factor = 2 if xdim == 2463 else 4
    # make an image whose pixel value corresonds to the radius from the center.
    # and this will be used to create on-the-fly beamstop masks of varying radius
    Y,X = np.indices((ydim, xdim))
    if jid==0:
        print("Beginning simulations...")

    # instantiate the simulator class
    HS = Simulator(DET, BEAM, cuda=not args.cpuMode,
                   verbose=args.verbose and jid==0)
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
    outname = os.path.join(args.outdir, "%s%d.h5" %(prefix,jid))
    if jid==0:
        cmd = os.path.join(args.outdir, "commandline.txt")
        config = open(paths_and_const.__file__, 'r').read()
        with open(cmd, "w") as o:
            o.write("working dir: %s\n" % os.getcwd())
            o.write("Python command: " + " ".join(sys.argv) + "\n")
            o.write("\nConfiguration (paths_and_const.py):\n%s" % config)

    with h5py.File(outname, "w") as out:
        out.create_dataset("nominal_mask", data=mask)
        ds_shape = 512,512
        if args.centerCrop:
            ds_shape = cropdim, cropdim
        comp_args = {"dtype": np.float32}

        if not args.noCompress:
            comp_args["compression_opts"] = 4
            comp_args["compression"] = "gzip"
            comp_args["shuffle"] = True
            comp_args["dtype"] = np.uint16
        dset_shape = (Nshot,) + ds_shape
        chunks = (1,)+ds_shape
        if args.shotsPerEx > 1:
            dset_shape = (Nshot, args.shotsPerEx) + ds_shape
            chunks = (1, args.shotsPerEx)+ds_shape
        dset = out.create_dataset("images",
                                  shape=dset_shape,
                                  chunks=chunks,
                                  **comp_args)

        comp_args.pop("dtype")
        cbf_names = []

        param_names = ["reso", "one_over_reso",
                       "radius", "one_over_radius",
                       "is_multi", "multi_lat_angle_sigma",
                       "num_lat", "bg_scale",
                       "beamstop_rad", "detdist", "wavelen",
                       "beam_center_fast", "beam_center_slow",
                       "cent_fast_train", "cent_slow_train",
                       "Na", "Nb", "Nc", "pdb", "mos_spread","xtal_scale"] \
                      + ["r%d" % x for x in range(1, 10)] + ['pitch_deg', 'yaw_deg', "bg_only"]
        geom_names = ["detdist", "wavelen", "pixsize", "xdim", "ydim"]
        lab_dset = out.create_dataset("labels", dtype=np.float32, shape=(Nshot, len(param_names)) , **comp_args)
        geom_dset = out.create_dataset("geom", dtype=np.float32, shape=(Nshot, len(geom_names)), **comp_args)
        lab_dset.attrs["names"] = param_names
        lab_dset.attrs["pdbmap"] = list(PDB_MAP)
        geom_dset.attrs["names"] = geom_names

        # list of rotation matrices (length is Nshot)
        if args.randAxis:
            assert gvec is not None
            angle= np.random.uniform(-180,180,Nshot)
            rot_vecs = np.array([gvec / np.linalg.norm(gvec)]*Nshot)
            rot_vecs *= angle[:,None]
            rotMats = Rotation.from_rotvec(rot_vecs, degrees=True).as_matrix()
        elif args.axisRotOnly is not None:
            angle = np.random.uniform(-180,180,Nshot)
            rot_vecs = np.zeros((Nshot, 3))
            rot_vecs[:,args.axisRotOnly] = angle
            rotMats = Rotation.from_rotvec(rot_vecs, degrees=True).as_matrix()
        elif args.twoAxisOnly is not None:
            angle = np.random.uniform(-180,180, Nshot)
            gvecs = np.random.normal(0,1,(Nshot, 2))
            uvecs = gvecs / np.linalg.norm(gvecs, axis=1)[:,None]
            #rot_vecs = uvecs*angle
            rot_vecs = np.zeros((Nshot, 3))
            if args.twoAxisOnly==0: # "xy"
                rot_vecs[:,[0,1]] = uvecs
            elif args.twoAxisOnly==1: # xz
                rot_vecs[:,[0,2]] = uvecs
            else:  # yz
                rot_vecs[:,[1,2]] = uvecs
            rot_vecs *= angle[:,None]
            rotMats = Rotation.from_rotvec(rot_vecs, degrees=True).as_matrix()
        else:
            rotMats = Rotation.random(Nshot).as_matrix()
        times = []  # store processing times per shot

        # random generators
        random_dist = random_wave = None
        if args.randDist:
            if args.randDistChoice is not None:
                random_dist = lambda: np.random.choice(args.randDistChoice)
            else:
                d1,d2 = args.randDistRange
                assert d1 < d2
                random_dist = lambda: np.random.uniform(d1,d2)
        if args.randWave:
            en1, en2 = args.randWaveRange
            assert en1 < en2
            random_wave = lambda: np.random.uniform(en1, en2)

        for i_shot in range(Nshot):
            t = time.time()
            cbf_name = None
            if args.saveRaw:
                cbf_dir = os.path.join(args.outdir, "cbfs%d" % jid)
                if not os.path.exists(cbf_dir):
                    os.makedirs(cbf_dir)
                cbf_name = os.path.join(cbf_dir, "shot_1_%05d.cbf" % i_shot)
                cbf_names.append(os.path.abspath(cbf_name))

            pdb_name = args.pdbName
            if pdb_name is not None:
                pdb_name = pdb_name.replace("//","/")

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

            HS.mask = shot_mask
            if not args.bgOnly and args.randHits:
                HS.bg_only = np.random.choice([0,1])

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
                                      cbf_name=cbf_name)

            if args.sanityTestOps:
                assert args.shotsPerEx == 1
                pdb_name = params['pdb_name']
                pdb_id = os.path.basename(pdb_name)
                OPS = np.load(paths_and_const.SGOP_FILE, allow_pickle=True)[()][pdb_id]
                print(pdb_id)
                assert paths_and_const.FIX_RES

                for i_op, U_o in enumerate(OPS):
                    rot2 = np.dot(rotMats[i_shot], np.reshape(U_o, (3,3)))
                    print("Doing op %d / %d" %(i_op+1, len(OPS)))
                    print(U_o)
                    params2, spots2, imgs2, _, _ = HS.simulate(rot_mat=rot2,
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
                                                     uniform_reso=args.uniReso)
                    if not np.allclose(spots, spots2):
                        assert args.centerCrop  # we only care about this allclose test if center crop is true (orientation mode)
                        max_pool = counter_utils.mx_gamma(stride=center_ds_fact, dim=cropdim)
                        ds_spots = []
                        for sp_img in [spots, spots2]:
                            ds_sp = counter_utils.process_image(sp_img, max_pool, useSqrt=True)[0]
                            IMAX = np.sqrt(65535)
                            ds_sp[ds_sp > IMAX] = IMAX
                            ds_sp = ds_sp.numpy().astype(np.uint16)
                            ds_spots.append(ds_sp)
                        assert np.allclose(ds_spots[0], ds_spots[1])
                exit()

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
                    img_1d[hot_inds] = 2**16
                    img = img_1d.reshape(img.shape)
                    img *= shot_mask
                    imgs[i_img] = img

            # add bad pixels
            if args.noBad:
                min_npix = int(0.01 * xdim)
                max_npix = 3*min_npix
                nbad = np.random.randint(min_npix, max_npix)
                bad_inds = np.random.permutation(npix)[:nbad]

                for i_img, img in enumerate(imgs):
                    img_1d = img.ravel()
                    img_1d[bad_inds] = 0
                    img = img_1d.reshape(img.shape)
                    img *= shot_mask
                    imgs[i_img] = img

            if paths_and_const.LAUE_MODE:
                ave_pool = counter_utils.mx_gamma(stride=center_ds_fact, use_mean=True)
                #ds_wavelen = counter_utils.process_image(params['wavelen_data'],
                #                                         ave_pool, useSqrt=False)[0]

            # Rules for downsampling
            if args.centerCrop:
                max_pool = counter_utils.mx_gamma(stride=center_ds_fact, dim=cropdim)
                dx = xdim *.5 / center_ds_fact - cropdim*.5
                dy = ydim *.5 / center_ds_fact - cropdim*.5
                # convert cent_x, cent_y to downsampled version
                cent_x_train = cent_x / center_ds_fact - dx
                cent_y_train = cent_y / center_ds_fact - dy
            else:
                max_pool = torch.nn.MaxPool2d(quad_ds_fact, quad_ds_fact)
                q = np.random.choice(["A", "B", "C", "D"])
                # convert cent_x, cent_y to downsampled version
                # TODO update cent_x_train, cent_y_train
                cent_x_train = (cent_x - xdim*.5)/quad_ds_fact #factor
                cent_y_train = (cent_y - ydim*.5)/quad_ds_fact #factor

            ds_imgs = []
            for i_img, img in enumerate(imgs):

                if paths_and_const.PEAK_MODE:
                    spots_i = spots[i_img]
                    img = spots_i > np.percentile(spots_i,99.99)

                if args.centerCrop:
                    ds_img = counter_utils.process_image(img, max_pool, useSqrt=True)[0]
                else:
                    ds_img = to_tens(img, shot_mask, maxpool=max_pool, ds_fact=quad_ds_fact, quad=q)

                ds_imgs.append(ds_img)

            Na, Nb, Nc = params["Ncells_abc"]
            #r1,r2,r3,r4,r5,r6,r7,r8,r9 = params["Umat"]
            r1,r2,r3,r4,r5,r6,r7,r8,r9 = rotMats[i_shot].ravel()
            if HS.bg_only:
                r1=r2=r3=r4=r5=r6=r7=r8=r9=np.nan
                params["num_lat"] = 0
                params["reso"] = np.nan
                radius = np.nan
                params["multi_lattice"] = 0
                params["ang_sigma"] = np.nan
                Na = Nb = Nc = np.nan
                pdb = np.nan
                params["mos_spread"]=np.nan
                params["crystal_scale"]=np.nan
            else:
                pdb = PDB_MAP[params["pdb_name"]]
            param_arr = [params["reso"], 1/params["reso"],
                 radius/quad_ds_fact, quad_ds_fact/radius, # TODO update depending on args.centerCrop?
                 params["multi_lattice"],
                 params["ang_sigma"],
                 params["num_lat"],
                 params["bg_scale"],
                 beamstop_rad,
                 params["detector_distance"],
                 params["wavelength"],
                 cent_x, cent_y,
                 cent_x_train, cent_y_train,
                 Na, Nb, Nc, 
                 pdb,
                 params["mos_spread"],
                 params["crystal_scale"],
                 r1,r2,r3,r4,r5,r6,r7,r8,r9,
                 params['pitch_deg'], params['yaw_deg'],
                 1 if HS.bg_only else 0]

            geom_array = [params["detector_distance"],
                             params["wavelength"],
                             pixsize,
                             xdim, ydim]

            if not args.noCompress:
                IMAX=np.sqrt(65535)
                for i_ds_img, ds_img in enumerate(ds_imgs):
                    ds_img[ds_img > IMAX] = IMAX
                    ds_img = ds_img.numpy().astype(np.uint16)
                    ds_imgs[i_ds_img] = ds_img

            if args.shotsPerEx == 1:
                assert len(ds_imgs)==1
                dset[i_shot] = ds_imgs[0]
            else:
                dset[i_shot] = ds_imgs
            geom_dset[i_shot] = geom_array
            lab_dset[i_shot] = param_arr

            t = time.time()-t
            times.append(t)
            print(f"RANK {jid+1}/{njobs}: Done with shot {i_shot+1}/{Nshot} out of {args.nshot} total (took {t:.4f} sec).", flush=True)

        ave_t = np.mean(times)
        print(f"RANK{jid+1}: Done! Takes {ave_t:.4f} sec on average per image. (Other processes might still be simulating)" % np.mean(times))

        out.attrs["cbf_names"] = cbf_names

