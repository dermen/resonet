# coding: utf-8

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("datadir", help="Path to folder containing validation mccd or cbf images. "
                                    "The dirname is usually of the format,  e.g. 1.90A", type=str)
parser.add_argument("modelname", help="path to the .nn model", type=str)
parser.add_argument("outfile", type=str, help="name of the output file that will be created")
parser.add_argument("--arch", type=str, choices=["res50", "res18", "res34", "le"], default="res50",
                    help="architecture of model (default: res50)")
parser.add_argument("--loop", action="store_true")
parser.add_argument("--display", action="store_true")
parser.add_argument("--savefig", action="store_true")
parser.add_argument("--figdir", default=None, type=str,
                    help="A directory for PNG files to be written to. "
                         "Default: tempfigs_X where X is a string representing resolution")
parser.add_argument("--maskFile", type=str, default=None)
parser.add_argument("--rawRadius", action="store_true", help="if predictor is rad and its the raw image radius (as opposed to downsampled image radius)")
parser.add_argument("--gpus", action="store_true")
parser.add_argument("--leaveOnGpu", action="store_true")
parser.add_argument("--quads", nargs="+", choices=["A", "B", "C", "D"], default="A", help="which quad to use for prediction")
parser.add_argument("--maxProc", type=int, default=None)
parser.add_argument("--ndev", type=int, default=1)
args = parser.parse_args()

from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD
import time

import numpy as np
import dxtbx
import glob
import re
import os
from resonet.utils.eval_model import load_model, raw_img_to_tens_pil, raw_img_to_tens
from resonet.utils import eval_model
from resonet.sims.simulator import reso2radius
import pylab as plt
from scipy.ndimage import binary_dilation
import torch
from resonet.utils import mpi
import fabio

if COMM.rank==0:print("Wil predict using quads", args.quads)

def B_from_d(d):
    return 4*d**2 + 12

def d_to_dnew(d):
    B = B_from_d(d)

    # new equation: B = 13*dnew^2 -23 *dnew + 29
    # quadratic fit coef
    a,b,c = 13., -22.,  26.-B
    dnew = .5* (-b + np.sqrt(b**2 - 4*a*c)) / a  # positive root
    return dnew

def sanitize_inputs(fnames):
    good_fnames = []
    for i,f in enumerate(fnames):
        if i % COMM.size != COMM.rank:
            continue
        try:
            loader = dxtbx.load(f)
            good_fnames.append(f)
        except KeyError:
            continue

        if COMM.rank==0:
            print("Verifying shot %d/ %d" % (i, len(fnames)))
    good_fnames = COMM.bcast(COMM.reduce(good_fnames))
    return good_fnames

def main():
    real_data_dirname = args.datadir
    assert os.path.isdir(real_data_dirname)
    MODEL_FNAME = args.modelname
    model = load_model(MODEL_FNAME, arch=args.arch)

    dev = "cpu"
    if args.gpus:
        gpu_id = mpi.get_gpu_id_mem(args.ndev)
        #gpu_id = COMM.rank % args.ndev
        dev = "cuda:%d" % gpu_id
        model = model.to(dev)
        #if COMM.rank==0:
        #    dev = "cuda:0"
        #    model = model.to(dev)
        #elif COMM.rank==1:
        #    dev = "cuda:1"
        #    model = model.to(dev)
    devs = COMM.gather(dev)
    if COMM.rank==0:
        from collections import Counter
        print("Device assignments: ")
        print(Counter(devs).items())

    fnames = glob.glob(real_data_dirname + "/*[0-9].cbf")
    if not fnames:
        fnames = glob.glob(real_data_dirname + "/*[0-9].mccd")
        # TODO add tensor conversion for MCCD files...

    import re
    temp = []
    for i,f in enumerate(fnames):
        if i % COMM.size != COMM.rank:continue
        if re.search("_[0-9]_[0-9]{5}" , f) is None:
            continue
        temp.append(f)
        if COMM.rank==0:
            print("done verifying image %d / %d" % (i+1, len(fnames)))
    fnames = COMM.bcast(COMM.reduce(temp))


    #fnames = sanitize_inputs(fnames)
    assert fnames

    def res_from_name(name):
        res = re.findall("[0-9]\.[0-9]+A", name)
        assert len(res)==1
        res = float(res[0].split("A")[0])
        return res

    target_res = res_from_name(real_data_dirname)


    loader = None
    all_t = []
    for i in range(len(fnames)):
        try:
            loader = dxtbx.load(fnames[i])
            B = loader.get_beam()
            D = loader.get_detector()
            detdist = abs(D[0].get_distance())
            wavelen = B.get_wavelength()
            pixsize = D[0].get_pixel_size()[0]

            #imgs = []
            #for i,f in enumerate(fnames[:20]):
            #    if i% COMM.size != COMM.rank:
            #        continue
            #    if COMM.rank==0:
            #        print("Inspecting for hot pixels... %d/%d" %(i, 100), flush=True)
            #    img = dxtbx.load(f).get_raw_data().as_numpy_array().astype(np.float32)
            #    img[img > 1e4] = 0
            #    imgs.append(img)
            #imgs = COMM.reduce(imgs)
            #hotpix = None
            #if COMM.rank==0:
            #    print("median...", flush=True)
            #    img_med =np.median(imgs, 0)
            #    hotpix = img_med > 1e2
            #hotpix = COMM.bcast(hotpix)

            xdim, ydim = D[0].get_image_size()

            is_pil = xdim==2463
            break
        except:
            pass

    assert loader is not None

    if args.maskFile is None:
        mask = loader.get_raw_data().as_numpy_array() >= 0
        mask = ~binary_dilation(~mask, iterations=1)
        beamstop_rad = 50
        Y,X = np.indices((ydim, xdim))
        R = np.sqrt((X-xdim/2.)**2 + (Y-ydim/2.)**2)
        out_of_beamstop = R > beamstop_rad
        mask = np.logical_and(mask, out_of_beamstop)
    else:
        mask = np.load(args.maskFile)

    #mask = np.logical_and(mask, ~hotpix)

    Nf = len(fnames)
    if COMM.rank==0:
        print("Found %d fnames" % Nf)
    factor = 2 if is_pil else 4
    target_rad = reso2radius(target_res, DET=D, BEAM=B) / factor
    if COMM.rank==0:
        print("Target res: %fA" % target_res)

    if args.savefig:
        figdir = args.figdir
        if figdir is None:
            figdir = args.outfile + ".tempfigs"
        if COMM.rank==0 and not os.path.exists(figdir):
            os.makedirs(figdir)
        COMM.barrier()

    rank_fnames = []
    rank_fignames = []
    all_hres = []
    all_resos = []
    rads = []
    IM = None

    maxpool = None #torch.nn.MaxPool2d(factor,factor)
    ttotal = time.time()
    fnames_to_proc = fnames
    if args.maxProc is not None:
        fnames_to_proc = fnames[:args.maxProc]
    for i_f, f in enumerate(fnames_to_proc):
        if i_f % COMM.size != COMM.rank: continue
        t = time.time()
        try:
            #loader = dxtbx.load(f)
            img = fabio.open(f).data
        except:
            continue
        #img = loader.get_raw_data().as_numpy_array()
        img = img.astype(np.float32)
        tread = time.time()-t
        #if is_pil:
        tens_getter = eval_model.raw_img_to_tens_pil2#3
        kwargs ={}
        kwargs["leave_on_gpu"] = args.leaveOnGpu
        kwargs["dev"] = dev
        kwargs["maxpool"] = maxpool
        kwargs["ds_fact"] = factor
        kwargs["cent"] = xdim/2.,ydim/2.
        #else:
        #    tens_getter = eval_model.raw_img_to_tens
        #    kwargs = {}
        
        geom = torch.tensor([[detdist, pixsize, wavelen, xdim, ydim]])
        geom = geom.to(dev)

        if args.loop:
            resos = []
            tds = 0
            teval = 0
            for quad in args.quads:
                tds_temp = time.time()
                tens = tens_getter(img, mask, quad=quad, **kwargs)
                if dev != "cpu" and tens.device.type != "cuda":
                    tens = tens.to(dev)
                if args.gpus:
                    torch.cuda.synchronize(device=dev)
                tds += time.time()-tds_temp

                teval_temp = time.time()
                pred = model(tens, geom)
                if args.gpus:
                    torch.cuda.synchronize(device=dev)
                reso = 1./pred.item()
                resos.append(reso)
                teval += time.time()-teval_temp
            tds = tds / len(args.quads)
            teval = teval / len(args.quads)

        else:
            tensors = []
            tds = time.time()
            for quad in args.quads:

                tens = tens_getter(img, mask, quad=quad, **kwargs)
                if dev != "cpu" and tens.device.type != "cuda":
                    tens = tens.to(dev)
                tensors.append(tens)
            if args.gpus: # and COMM.rank in [0,1]:
                torch.cuda.synchronize(device=dev)
            tds = time.time()-tds

            teval = time.time()
            tensors = torch.concatenate(tensors)
            pred = model(tensors, geom)
            if args.gpus:
                torch.cuda.synchronize(device=dev)
            resos = [1/r.item() for r in pred]
            teval = time.time() - teval

        all_resos.append(resos)
        #resos = [d_to_dnew(r) for r in resos]
        #res_rads = list(zip(resos, radii, args.quads))
        #res_rads.append( (res, radius, i_tens ))
        #res, radius, i_tens = sorted(res_rads)[0]
        hres = np.min(resos)
        radius = reso2radius(hres, D, B) / factor
        #res = d_to_dnew(res)
        #teval = time.time()-teval

        #print(radius, target_rad, i_tens)
        print("%.3f" %hres, target_res,  "rd=%.4f, ds=%.4f, ev=%.4f" % (tread, tds, teval)) #radius, target_rad, i_tens)
        all_t.append([tread, tds, teval])
        all_hres.append(hres)
        rads.append(radius)
        rank_fnames.append(f)
        if args.display:
            new_data = tensors[0].numpy()[0,0,:512,:512]
            C2 = plt.Circle(xy=(0,0), radius=radius, ec='r', ls='--', fc='none' )
            if IM is None:
                IM = plt.imshow(new_data, vmax=10)
                Cref = plt.Circle(xy=(0,0), radius=target_rad, ec='w', ls='--', fc='none' )
                plt.gca().add_patch(Cref)
                plt.gca().add_patch(C2)
            else:
                IM.set_data(new_data)
                plt.gca().patches.pop()
                plt.gca().add_patch(C2)
            #plt.cla()
            #plt.imshow(tens.numpy()[0, 0, :512, :512], vmax=10)

            plt.title("%.2fA: %s" % (target_res, os.path.basename(f)))
            if args.savefig:
                figname = os.path.join( figdir, "%.2fA_%05d.png" % (target_res, i_f))
                plt.savefig(figname, dpi=150)
                rank_fignames.append(figname)

            if COMM.rank==3:
                plt.draw()
                plt.pause(0.2)

    COMM.barrier()
    ttotal = time.time()-ttotal
    all_t_all_rank = COMM.reduce(all_t)
    all_t = [(COMM.rank,) + tuple(times) for times in all_t]
    all_t_per_rank = COMM.reduce(all_t)
    if COMM.rank==0:
        tread, tds, tev =  map( np.mean, zip(*all_t_all_rank))
        n = len(all_t_all_rank)  # total images
        print("Elapsed time: %.3f sec" % (ttotal,))
        print("Per-rank time per image: %.4f sec (rd=%.4f sec, ds=%.4f sec, ev=%.4f sec)" % (tread+tds+tev,tread, tds, tev))
        #print("Effective rank time per image: %.4f sec (rd=%.4f, ds=%.4f, ev=%.4f)" % (ttotal/n,tread/n, tds/n, tev/n))
        print("Elapsed real time per image: %.4f sec" % (ttotal / n ))
        print("Effective processing rate %.4f Hz" % (n / ttotal))

    rads = COMM.reduce(rads)
    all_hres = COMM.reduce(all_hres)
    all_resos = COMM.reduce(all_resos)
    fnames = COMM.reduce(rank_fnames)
    if args.savefig:
        rank_fignames = COMM.reduce(rank_fignames)


    if COMM.rank==0:
        all_hres = np.array(all_hres)
        order = np.argsort(rads)[::-1]
        if args.savefig:
            ordered_figs = [rank_fignames[i] for i in order]
            for i, f in enumerate(ordered_figs):
                new_f = os.path.join(figdir, "sorted_%05d.png" % i)
                if os.path.exists(new_f):
                    os.remove(new_f)
                os.symlink(os.path.abspath(f), new_f)

        perc10 = np.percentile(all_hres,10)
        max_top10 = all_hres[all_hres <= perc10].mean()
        s = real_data_dirname + "Res: %.4f +- %.4f (Angstrom). highest= %.4fA . MeanHighest10perc=%.4fA (detdist=%.2f mm)" \
            % (np.mean(all_hres), np.std(all_hres), np.min(all_hres), max_top10, detdist)
        print(s)
        #from IPython import embed;embed()
        np.savez(args.outfile, rads=rads, fnames=fnames, pixsize=pixsize, detdist=detdist, wavelen=wavelen, res=all_hres, all_resos=np.array(all_resos),
                 result_string=s, factor=factor, fignames=rank_fignames, target_rad=target_rad, target_res=target_res, all_t_per_rank=all_t_per_rank)
        o = args.outfile
        if not o.endswith(".npz"):
            o = o + ".npz"
        print("Saved file %s" % o)
        print("Done.")


if __name__=="__main__":
    main()

