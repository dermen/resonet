from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("expt", type=str)
ap.add_argument("modelpath", type=str)
ap.add_argument("out", type=str)
ap.add_argument('--cutoff', default=0.9, type=float)
ap.add_argument("--ndev", type=int, default="1")
ap.add_argument("--format", choices=["coo","2d"], default="coo", type=str)
ap.add_argument("--compression", choices=["lzf", "gzip"], type=str, default="gzip")
ap.add_argument("--maximgs", type=int, default=None)
args = ap.parse_args()
assert 0 < args.cutoff < 1

from mpi4py import MPI
COMM = MPI.COMM_WORLD
from dxtbx.model import ExperimentList
import numpy as np
from resonet.utils.multi_panel import split_eiger_16M_to_panels
import torch
from resonet.compress import compress_models, data_format

def print0(*args, **kwargs):
    if COMM.rank==0:
        print(*args, **kwargs)

dev = "cpu"
if args.ndev > 0:
    dev_id = COMM.rank  % args.ndev
    dev = "cuda:%d"% dev_id

if COMM.rank < COMM.size:
    model = compress_models.load_model(args.modelpath)
    model = model.float().to(dev)

print0("Loading expt")
El = ExperimentList.from_file(args.expt)
print0("Done.")
iset = El[0].imageset
if args.maximgs is not None:
    iset = iset[:args.maximgs]
scan = El[0].scan
scan.set_image_range((1,len(iset)))
gonio = El[0].goniometer
det = El[0].detector
beam = El[0].beam
temp_img = iset.get_raw_data(0)[0].as_numpy_array()

_, _, _, panels, multi_panel_det = split_eiger_16M_to_panels(temp_img, det)

fops = {}

if args.compression=="lzf":
    comps={"compression":"lzf","shuffle":True}
else:
    comps={"compression":"gzip", "compression_opts":4, "shuffle":True}


if COMM.rank==COMM.size-1:
    if args.format=="coo":
        h5 = data_format.DiffCompWriter(args.out, detector=multi_panel_det,
                                        beam=beam, dtype=np.float16,
                                        compression_args=comps,
                                        scan=scan, goniometer=gonio, file_ops=fops)
    else:
        from simtbx.nanoBragg import utils
        num_images = len(iset)
        ydim, xdim = panels[0].shape
        img_shape = len(multi_panel_det), ydim, xdim
        h5 = utils.H5AttributeGeomWriter(args.out, img_shape, num_images,
                                         multi_panel_det, beam, dtype=np.float16,
                                         compression_args=comps,
                                         goniometer=gonio, scan=scan)
    nexits = 0
    while 1:
        message = COMM.recv()
        if isinstance(message, str):
            nexits += 1
            print("Received exit ; nexits total=%d" %  nexits)

        elif args.format=="coo" and isinstance(message, list) and len(message)==5:
            img_name, pid, slow, fast, val = message
            print("Received data for %s;  writing" % img_name)
            h5.add_image(pid=pid, fast=fast, slow=slow, val=val, key=img_name)

        elif args.format=="2d" and isinstance(message, list) and len(message)==2:
            idx, panels = message
            print("Received data for writing, img idx=%d" % idx)
            h5.add_image(panels)
        else:
            print("Unknown message type:", message)
        if nexits == COMM.size-1:
            break
    print("Closing file...")
    h5.close_file()

else:
    #print0("Found %d images!" % len(iset))
    sent_req = []
    for i_img in range(len(iset)):
        if i_img % (COMM.size-1) != COMM.rank:
            continue
        #print0("Compressing block of images %d/%d" % (i_img+1, len(iset)))
        img_name = "image%d" % i_img
        img = iset.get_raw_data(i_img)[0].as_numpy_array()

        _,_,_,panels = split_eiger_16M_to_panels(img)
        panel_peaks = []
        for i_pan, p in enumerate(panels):
            p = torch.tensor(p[None,None]).float().to(dev)
            out = model(p)
            peaks = (out > args.cutoff)[0,0].detach().cpu().numpy()
            panel_peaks.append(peaks)
        panel_peaks = np.array(panel_peaks)
        panels = np.array(panels)

        if args.format=="coo":
            pid, slow, fast = np.where(panel_peaks)
            val = panels[pid, slow, fast]
            req = COMM.isend([img_name, pid, slow, fast, val], dest=COMM.size-1)
        else:
            panels[~panel_peaks] = 0
            req = COMM.isend([i_img, panels], dest=COMM.size-1)
        sent_req.append(req)
    req = COMM.isend("EXIT", dest=COMM.size-1)
    sent_req.append(req)
    for req in sent_req:
        req.wait()
