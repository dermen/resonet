from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("-i", "--image",  required=True, type=str)
ap.add_argument("-m","--model-path",  dest="modelpath", default=None, help="optional model path, if not, default sparsification model will be loaded", type=str)
ap.add_argument("-o", "--outdir",required=True ,type=str)
ap.add_argument('--cutoff', default=0.9, type=float)
ap.add_argument("--ndev", type=int, default="1")
ap.add_argument("--format", choices=["coo","2d"], default="coo", type=str)
ap.add_argument("--compression", choices=["lzf", "gzip", "none"], type=str, default="gzip")
ap.add_argument("--closings", type=int, default=2)
ap.add_argument("--openings", type=int, default=4)
ap.add_argument("--dilations", type=int, default=1)
ap.add_argument("--dialsMode", action="store_true", help="if True, skip the AI model and just use DIALS to find spots")
ap.add_argument("--dtype", default="float16",type=str, choices=["float16", "float32", "float64"] )
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()

assert 0 < args.cutoff < 1

import os
import numpy as np
import glob
import re
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing
import torch
from dxtbx.model.experiment_list import ExperimentListFactory
from mpi4py import MPI
COMM = MPI.COMM_WORLD
has_simtbx = False
try:
    from simtbx.nanoBragg.utils import H5AttributeGeomWriter
    has_simtbx = True
except (ImportError, ModuleNotFoundError):
    assert not args.format=="2d"


from resonet.utils.multi_panel import split_eiger_16M_to_panels
from resonet.sparsify import sparsify_models, data_format
from resonet.sparsify import find_spots


def vprint(*print_args, **print_kwargs):
    if args.verbose:
        print(*print_args, **print_kwargs)


class Writer:
    def __init__(self, outname, expt, args):
        if args.compression == "lzf":
            comps = {"compression": "lzf", "shuffle": True}
        elif args.compression == "gzip":
            comps = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
        else:
            comps = {}

        iset = expt.imageset
        scan = expt.scan
        scan.set_image_range((1, len(iset)))
        gonio = expt.goniometer
        det = expt.detector
        beam = expt.beam
        dummie_img = iset.get_raw_data(0)[0].as_numpy_array()

        _, _, _, _, multi_panel_det = split_eiger_16M_to_panels(dummie_img, det)
        self.h5 = None
        """h5 is the file handle for either the COO format or the AttributeGeom format"""
        if args.format == "coo":
            self.h5 = data_format.DiffCompWriter(outname, detector=multi_panel_det,
                                            beam=beam, compression_args=comps,
                                            scan=scan, goniometer=gonio)
        else:
            num_images = len(iset)
            panel_xdim, panel_ydim = multi_panel_det[0].get_image_size()
            img_shape = len(multi_panel_det), panel_ydim, panel_xdim
            # TODO: double check DTYPE
            self.h5 = H5AttributeGeomWriter(outname, img_shape, num_images,
                                             multi_panel_det, beam, dtype=args.dtype,
                                             compression_args=comps,
                                             goniometer=gonio, scan=scan)
        self.nexits = 0
        """used to track the number of exit messages received by sparsify workers
        """
        self.args = args
        """command line arguments"""
        vprint("Writer initialized.")

    def process_message(self, message):
        """sparsify workers will send messages either containing sparsified data, or exit messages that
        indicate they have completed all of their assigned work"""
        if isinstance(message, str):
            self.nexits += 1
            vprint("Writer received exit ; nexits total=%d" % self.nexits)

        elif self.args.format == "coo" and isinstance(message, list) and len(message) == 5:
            img_idx, pid, slow, fast, val = message
            vprint("Writer received data for image %s;  writing COO format" % img_idx)
            self.h5.add_image(pid=pid, fast=fast, slow=slow, val=val, scan_num=img_idx)

        elif self.args.format == "2d" and isinstance(message, list) and len(message) == 2:
            img_idx, panels = message
            vprint("Writer received data for image %s;  writing 2d format" % img_idx)
            self.h5.add_image(panels)
        else:
            vprint("Unknown message type:", message)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        vprint("Writer closing file...")
        self.h5.close_file()
        print(f"Writer (rank={COMM.rank}) done closing file!")


def sparsify_image(img, model, dev, args):
    _, _, _, panels = split_eiger_16M_to_panels(img)
    panel_peaks = []

    for i_pan, p in enumerate(panels):
        if not args.dialsMode:
            p = torch.tensor(p[None, None]).float().to(dev)
            out = model(p)
            peaks = (out > args.cutoff)[0, 0].detach().cpu().numpy()
        else:
            peaks = find_spots.dials_find_spots(p, sigma_background=2, sigma_strong=1,
                                                algorithm="dispersion_extended")
        if args.closings > 0:
            peaks = binary_closing(peaks, iterations=args.closings)
        if args.dilations > 0:
            peaks = binary_dilation(peaks, iterations=args.dilations)

        # TODO make this tunable:
        # the idea here is to get rid of small peaks with erosion followed by dilation of same amount
        if args.openings > 0:
            peaks = binary_dilation(binary_erosion(peaks, iterations=args.openings), iterations=args.openings)

        panel_peaks.append(peaks)
    panel_peaks = np.array(panel_peaks)
    panels = np.array(panels)
    return panels, panel_peaks


def sparsify_expt(expt, args, outname):
    dev = "cpu"
    if args.ndev > 0:
        dev_id =COMM.rank % args.ndev
        dev = "cuda:%d" % dev_id

    if COMM.rank == COMM.size - 1:
        print("Yay Im a writer!")
        with Writer(outname, expt, args) as writer:
            while 1:
                message = COMM.recv()
                writer.process_message(message)

                if writer.nexits == COMM.size - 1:
                    break

    else:
        model = sparsify_models.load_model(args.modelpath)
        model = model.float().to(dev)
        sent_req = []
        iset = expt.imageset
        for i_img in range(len(iset)):
            if i_img % (COMM.size - 1) != COMM.rank:
                continue
            vprint(f"Worker {COMM.rank} processing image {i_img+1}/{len(iset)}", flush=True)
            img = iset.get_raw_data(i_img)[0].as_numpy_array()
            panels, panel_peaks = sparsify_image(img, model, dev, args)
            if args.format == "coo":
                pid, slow, fast = np.where(panel_peaks)
                val = panels[pid, slow, fast]
                req = COMM.isend([i_img, pid, slow, fast, val], dest=COMM.size - 1)
            else:
                panels[~panel_peaks] = 0
                req = COMM.isend([i_img, panels], dest=COMM.size - 1)
            sent_req.append(req)
        vprint("Worker %d exiting" % COMM.rank, flush=True)
        req = COMM.isend("EXIT", dest=COMM.size - 1)
        sent_req.append(req)
        for req in sent_req:
            req.wait()
        vprint("Worker %d Done!" % COMM.rank)
    COMM.barrier()


def exptlist_from_imgname(imgname, outexpt):
    # TODO assert image ends with _00001.cbf or similar
    patt = "_[0-9]{5}.cbf"
    assert re.search(patt, imgname) is not None, "image name must end in %05d.cbf pattern"
    imgname_glob = re.split(patt, imgname)[0] + "*.cbf"
    all_imgnames = glob.glob(imgname_glob)
    print("Createing exptlist from %d files"  % (len(all_imgnames)))
    El = ExperimentListFactory.from_filenames(filenames=all_imgnames)
    El.as_file(outexpt)
    print(f"Wrote experiment list to disk {outexpt}")


outexpt =os.path.join(args.outdir, "imported_for_sparse.expt")
if COMM.rank==0:
    os.makedirs(args.outdir, exist_ok=True)
    exptlist_from_imgname(args.image, outexpt)
COMM.barrier()
from dxtbx.model import ExperimentList
Elst = ExperimentList.from_file(outexpt)

for i_expt, expt in enumerate(Elst):
    if COMM.rank==0:
        print("Loading expt %d / %d (iset=%d images)" % (i_expt+1, len(Elst), len(expt.imageset)), flush=True)
    outname = os.path.join(args.outdir, "sparse_%d_master.h5" % (i_expt+1))
    sparsify_expt(expt, args, outname)
