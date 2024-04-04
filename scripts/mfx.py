

# Install Resonet
# https://smb.slac.stanford.edu/~resonet/#basic

# DOWNLOAD THESE MODELS
#wget https://smb.slac.stanford.edu/~resonet/overlapping.nn  # archstring=res34
#wget https://smb.slac.stanford.edu/~resonet/resolution.nn  #archstring=res50
#wget https://smb.slac.stanford.edu/~resonet/reso_retrained.nn  #archstring=res50
reso_model = "resolution.nn" # use for shorter wavelengths (~1 Ang), distances 200-300 mm
reso_model = "reso_retrained.nn" # use for longer wavelengths (~1.3 Ang), distances 60-90 mm
multi_model = None#"overlapping.nn"

# See https://www.rayonix.com/product/mx340-xfel/
rayonix_unbinned_size = 7680 # number of unbinned pixels across
rayonix_dim_mm = 340 # Rayonix XFEL340 is 340 mm across

# detzOffset is calibrated for MFX from data collected March 2023
# beam center is calibrated for MFX from data collected March 2023
# the script should be run on ampere cluster with --gpus-per-node=4 --cpus-per-gpu=2 for maximum throughput
# can be run on multiple nodes for increased throughput

def main():

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--detzOffset", default=-140.02, help="This value is added to the detector z encoder value in order to get the sample to detector distance (The default value here is -140.02, which is calibrated from experiment mfxl103222)", type=float)
    parser.add_argument("--nominalDetz", default=None, help="Provide a rough estimate of detector z in mm, in the event that the offset is unknown! Default=None meaning unused", type=float)
    parser.add_argument("--nominalWavelen", default=None, help="the photon wavelength in Angstrom (if None (default), the wavelength will be automatically extracted from XTC)", type=float)
    parser.add_argument("--aduPerPhoton", default=1, help="The conversion factor to go from ADUs to photons", type=float)
    parser.add_argument("--centerMM", default=[170.77, 169], help="the fast-scan, slow-scan coordinate of the forward beam on the image, in mm units (default=[170.77, 169] calibrated from experiment mxfl103222)", nargs=2, type=float)
    parser.add_argument("--ndevPerNode", help="Number of GPUs per compute node (default=4)", type=int, default=4)
    parser.add_argument("--run", help="run number (default=20)", type=int, default=20)
    parser.add_argument("--expt", help="experiment string (default=mfxl1032222)", default="mfxl1032222", type=str)
    parser.add_argument("--maxImg", type=int, default=None, help="Set to some integer to only process that many images")
    parser.add_argument("--rayonixAddr", help="psana DetName of the Rayonix (default=Rayonix)", type=str, default="Rayonix")
    parser.add_argument("--detzAddr", help="psana Detname of the detector z encoder (default=detector_z)", type=str, default="detector_z")
    args = parser.parse_args()
    detz_offset = args.detzOffset
    gain = args.aduPerPhoton
    cent_mm = args.centerMM
    nominal_detz = args.nominalDetz
    ndev_per_node = args.ndevPerNode
    xray_camera_addr = args.rayonixAddr
    detz_addr = args.detzAddr

    import os
    if not os.path.exists(reso_model):
        raise OSError("Please download resolution model using: wget https://smb.slac.stanford.edu/~resonet/reso_retrained.nn")
    #if not os.path.exists(multi_model):
    #    raise OSError("Please download overlapping lattice model using: wget https://smb.slac.stanford.edu/~resonet/overlapping.nn")
    import time
    import numpy as np
    import torch
    import socket
    from scipy import constants

    try:
        import psana
        from mpi4py import MPI
        COMM = MPI.COMM_WORLD
    except:
        print("Script requires psana and mpi4py")

    from resonet.utils.predict_fabio import ImagePredictFabio

    ds = psana.DataSource("exp=%s:run=%d:idx" % (args.expt, args.run))
    run = next(ds.runs())
    times = run.times()
    nevent = len(times)
    RAY = psana.Detector(xray_camera_addr)
    detz_encoder = psana.Detector(detz_addr)
    events = ds.events()

    dev = "cpu"
    if torch.cuda.is_available():
        dev = "cuda:%d" % (COMM.rank % ndev_per_node )
    print("RANK %d will use DEVICE %s on HOST %s" % (COMM.rank, dev, socket.gethostname()))

    P = ImagePredictFabio(
        reso_model=reso_model,
        multi_model=multi_model,
        ice_model=None,
        counts_model=None,
        reso_arch="res50",
        multi_arch="res34",
        ice_arch=None,
        counts_arch=None,
        dev=dev,
        use_modern_reso=True,
        B_to_d=None,
        )

    # read one image in order to determine binning size to measure ds_stride
    #if binning not in {1,2,3,4,5,6,8,10}:
    #    raise ValueError("binning mode not in 1,2,3,4,5,6,8,10")
    # map from binning mode -> maxpool downsample for resonet
    binning_map = {1:8, 2:4, 3:2, 4:1, 5:1, 6:1, 8:1, 10:1}

    P.quads = [-1] # -1 means to use a randomized quadrant for each inference
    P.gain = gain

    COMM.barrier()
    # when processing starts
    tstart = time.time()
    count = 0
    teid, twave, tdetz,tevent, tcalib , tload , tres = [],[],[],[],[],[],[]
    EBeam = psana.Detector("EBeam")
    en_convert = 1e10 * constants.c * constants.h / constants.electron_volt
    for i_ev, event_t in enumerate(times):
        t = time.time()
        ev = run.event(event_t)
        tevent.append(time.time()-t)
        if ev is None:
            continue
        if i_ev % COMM.size != COMM.rank:
            continue

        t = time.time()
        if args.nominalWavelen is None:
            try:
                ev_ebeam = EBeam.get(ev)
                energy = ev_ebeam.ebeamPhotonEnergy()
                assert energy > 0
                wavelen = en_convert/energy
            except (AttributeError, KeyError, AssertionError):
                print("Failed to extract wavelength from XTC! Provide a nominalWavelen value")
                continue
        else:
            wavelen = args.nominalWavelen
        twave.append( time.time()-t)
            
        t = time.time()
        img = RAY.calib(ev)
        tcalib_temp = time.time()-t
        if img is None:
            continue
        tcalib.append( tcalib_temp)
        ydim, xdim = img.shape # should be identical for rayonix,  ydim=xdim
        pixsize = rayonix_dim_mm / ydim
        binning = int(rayonix_unbinned_size / ydim) 
        if binning not in binning_map:
            continue
        
        P.cent = [x/pixsize for x in cent_mm]
        P.ds_stride = binning_map[binning]

        t = time.time()
        if args.nominalDetz is None:
            detz = detz_encoder(ev) + detz_offset
        else:
            detz = args.nominalDetz
        tdetz.append( time.time()-t)

        t = time.time()
        if not img.dtype==np.float32:
            img = img.astype(np.float32)

        P.load_image_from_file_or_array(detdist=detz, 
                pixsize=pixsize, wavelen=wavelen, raw_image=img)
        tload.append(time.time()-t)

        t = time.time()
        d = P.detect_resolution()
        tres.append( time.time()-t)

        #t = time.time()
        #p = P.detect_multilattice_scattering()
        #tmult = time.time()-t
        count += 1
        t = time.time()
        eid = ev.get(psana.EventId)
        sec, nsec = eid.time()
        fid=eid.fiducials()
        teid.append( time.time()-t)
        print(f"Resolution is {d:.2f} Ang. for image {i_ev+1}/{nevent} (sec={sec},nsec={nsec},fiducial={fid}).")
        if args.maxImg is not None and i_ev> args.maxImg:
            break

    count = COMM.reduce(count)
    ttotal = time.time()-tstart
    tcalib = COMM.reduce(tcalib)
    tload = COMM.reduce(tload)
    tres = COMM.reduce(tres)
    twave = COMM.reduce(twave)
    tdetz = COMM.reduce(tdetz)
    teid = COMM.reduce(teid)
    tevent = COMM.reduce(tevent)


    if COMM.rank==0:
        print(f"Processed {count} images in {ttotal:.2f} seconds ({count/ttotal:.2f} Hz)")
        print("Per-rank time to (medians)... ")
        print(f"... get Event: {np.median(tevent)*1000:.2f} millisec")
        print(f"... get Wavelength: {np.median(twave)*1000:.2f} millisec")
        print(f"... get Rayonix calibrated image from XTC: {np.median(tcalib)*1000:.2f} millisec")
        print(f"... get DetZ: {np.median(tdetz)*1000:.2f} millisec")
        print(f"... load Rayonix image to torch tensor: {np.median(tload)*1000:.2f} millisec")
        print(f"... estimate resolution: {np.median(tres)*1000:.2f} millisec")
        print(f"... get EventId timestamp: {np.median(teid)*1000:.2f} millisec")


if __name__=="__main__":
    main()

