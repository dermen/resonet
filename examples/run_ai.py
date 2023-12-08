
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("datadir", type=str, help="path to folder with image files of the form some_1_00001.*")
parser.add_argument("--resoModel", type=str, nargs=2, help="2 argument, first is the .nn file, second is the arch string")
parser.add_argument("--multiModel", type=str, nargs=2, help="2 argument, first is the .nn file, second is the arch string")
parser.add_argument("--outdir", type=str, help="output folder", default=None)
parser.add_argument("--quads",default=[1], type=int, nargs="+", help="quad indices (any combination of [0,1,2,3]. to use all 4 quads use --quads 0 1 2 3")
parser.add_argument("--gpu", action="store_true", help="whether to use a GPU")
parser.add_argument("--ext", type=str, default="cbf", help="file extension (default: cbf)")
parser.add_argument("--noMLP", action="store_true", help="if True, use the quadratic B-factor vs resolution model, otherwise uses the multi-layer perceptron model fit with Pytorch")
parser.add_argument("--makeMaxImgs", action="store_true", help="save max composite images of the highest/lowest resolution and overlapping lattice prediction images")
parser.add_argument("--aduPerPhot", type=float, default=1)
parser.add_argument("--maxProcess", type=int, default=None)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--nt", type=int, default=None)
args = parser.parse_args()


import os
import sys
import glob
import numpy as np
from resonet.utils import predict_dxtbx
import time
import dxtbx
from dials.array_family import flex
from dxtbx.model import Experiment, ExperimentList
from dials.command_line.find_spots import working_phil
from dials.algorithms.spot_finding import per_image_analysis
import pandas
if args.nt is not None:
    import torch
    print(torch.get_num_threads())
    torch.set_num_threads(args.nt)
from mpi4py import MPI

import pylab as plt


def get_num(fname):
    num = int(fname.split("_")[-1].split(".")[0])
    return num


def max_image(filenames):
    mx_img = None
    for i_f, f in enumerate(filenames):
        print("Max image %d / %d" % (i_f+1, len(filenames)), end="\r", flush=True)

        img = dxtbx.load(f).get_raw_data().as_numpy_array()
        if mx_img is None:
            mx_img = img
        else:
            mx_img = np.max([mx_img, img], axis=0)
    print("")
    return mx_img


def get_mx_imgs(df, by="ai_d", num=10):
    dsort = df.sort_values(by=by)
    high = dsort.iloc[:num]
    low = dsort.iloc[-num:]
    mx_high = max_image(high.filename)
    mx_low = max_image(low.filename)
    return mx_high, mx_low

class Plotter:
    def __init__(self):
        self.lines_fig, self.lines_ax = plt.subplots(ncols=1, nrows=3, layout='constrained')
        self.img_fig, self.img_ax = plt.subplots(ncols=1, nrows=1, layout='constrained')

    def init_plots(self):
        labs = ["$d$", "$p_i$", r"$N_{\rm spots}$"]
        for i_ax, ax in self.lines_ax:
            ax.set_ylabel(labs[i_ax])

    def update_image(self, img):
        sel = img > 0
        m = img[sel].mean()
        s = img[sel].std()
        vmax=m+s
        vmin=m-s
        if self.img_ax.images:
            self.img_ax.images[0].set_data(img)
            self.img_ax.images[0].set_clim(vmin, vmax)
        else:
            self.img_ax.imshow(img, vmin=vmin, vmax=vmax)

    def update_lines(self, data):
        # data is 3-tuple otf  resolution, overlapping lattice probability, peak counts
        assert len(data)==3
        if not self.lines_ax[0].lines:
            for i_ax, ax in enumerate(self.lines_ax):
                ax.plot([1], [data[i_ax]], 'o', ms=3)
        else:
            for i_ax, ax in enumerate(self.lines_ax):
                line = ax.lines[0]
                x, y = map(list,line.get_data())
                x.append(len(x))
                y.append(data[i_ax])
                line.set_data(x,y)
                xlim = min(x), max(x)
                ylim = min(y), max(y)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

    def update_display(self, pause=1e-3):
        self.lines_fig.canvas.draw_idle()
        plt.pause(pause)
        self.img_fig.canvas.draw_idle()
        plt.pause(pause)



def main():
    COMM = MPI.COMM_WORLD
    if args.outdir is not None and COMM.rank == 0 and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        cmdline = " ".join(sys.argv)
        with open(os.path.join(args.outdir, "commandline_input.txt"), "w") as o:
            o.write(cmdline + "\n")

    PLOTTER = None
    if args.plot:
        PLOTTER = Plotter()

    outname = args.datadir.replace("/","_")

    fnames = None
    if COMM.rank==0:
        try:
            fnames = glob.glob(os.path.join(args.datadir, "*_[1-9]_*%s" % args.ext))
            assert fnames
            key1 = lambda x: int(x.split("_")[-2])
            key2 = lambda x: int(x.split("_")[-1].split(".")[0])
            fnames = sorted(sorted(fnames, key=key1), key=key2)
        except:
            print("Sorting failed!")
            fnames = glob.glob(os.path.join(args.datadir, "*.%s" % args.ext))
    fnames = COMM.bcast(fnames)

    outputs = []
    reso_model, reso_arch = args.resoModel
    multi_model, multi_arch = args.multiModel
    multi_arch = "res34"
    counts_model = counts_arch = None
    B_to_d = None

    params = working_phil.extract()
    params.spotfinder.filter.ice_rings.filter = True

    if args.gpu:
        dev = "cuda:0"
    else:
        dev = "cpu"

    P = predict_dxtbx.ImagePredictDxtbx(
            reso_model=reso_model,
            reso_arch=reso_arch,
            multi_model=multi_model,
            multi_arch=multi_arch,
            counts_model=counts_model, 
            counts_arch=counts_arch, 
            dev=dev,
            B_to_d=B_to_d)

    P.quads = args.quads
    P.gain = args.aduPerPhot
    P.cache_raw_image = True
    start_time = time.time()
    all_tres, all_tcount, all_tmu = [],[],[]
    all_tproc = []
    tproc = 0
    for i_f, fname in enumerate(fnames):

        if args.maxProcess is not None and i_f >= args.maxProcess:
            continue
        
        if i_f % COMM.size != COMM.rank:
            continue
        bn = os.path.basename(fname)

        # predict with RESONET
        P.load_image_from_file(fname)
        t = time.time()
        d = P.detect_resolution()
        tres = time.time()-t
        t = time.time()
        counts = P.count_spots()
        tcount = time.time()-t
        t = time.time()
        mu = P.detect_multilattice_scattering(binary=False)
        tmu = time.time()-t
        all_tmu.append(tmu)
        all_tcount.append(tcount)
        all_tres.append(tres)
        if args.plot:
            PLOTTER.update_image(P.raw_image)
            PLOTTER.update_lines([d, mu, counts])
            #print("%s: (%d/%d). d=%.2f, Nref=%d, Multi=%.2f"
            #      % (bn, i_f+1, len(fnames), d,  counts, mu))
            elapsed = time.time()-start_time
            if elapsed > 5:
                PLOTTER.update_display()
                start_time = time.time()
        # end predict with RESONET
        #continue

        # run spot finder
        loader = dxtbx.load(fname)
        E = Experiment()
        E.imageset = loader.get_imageset([fname])
        E.detector = loader.get_detector()
        E.beam = loader.get_beam()
        E.scan = loader.get_scan()
        E.goniometer = loader.get_goniometer()
        El = ExperimentList()
        El.append(E)
        refls = flex.reflection_table.from_observations(El, params)
        refls.centroid_px_to_mm(El)
        refls.map_centroids_to_reciprocal_space(El)
        
        dials_d = -1
        try:
            dials_d = per_image_analysis.estimate_resolution_limit(refls)
        except Exception as err:
            pass
        dials_counts = len(refls)
        # END run spot finder

        # store results
        num = get_num(fname)
        outputs.append( (num, counts, dials_counts, d, dials_d, mu, os.path.abspath(fname), tres, tcount, tmu) )

        tproc += tres + tmu + tcount
        print("%s: (%d/%d). d=[%.2f VS %.2f], Nref=[%d VS %d], Multi=%.2f, times=%.3f, %.3f, %.3f (%s), cumm tproc=%.3f"
                % (bn, i_f+1, len(fnames), d, dials_d, counts, dials_counts, mu, tres, tcount, tmu, dev, tproc))
        all_tproc.append(tproc)

    outputs = COMM.reduce(outputs)
    all_tproc = COMM.reduce(all_tproc)

    if args.outdir is not None and COMM.rank==0:
        med = np.median(all_tproc)
        print("Tproc per rank = %.4f, effective rate=%.1f Hz" % (med, len(all_tproc)/med))
        if outname.endswith("_"):
            outname = outname[:-1]
        if outname.startswith("._"):
            outname = outname[2:]

        #fmt = "%d", "%d", "%d", "%.3f", "%.3f", "%.3f", "%s"
        cols = "img_num", "ai_nref", "dials_nref", \
               "ai_d", "dials_d", "multi", "filename", "tres", "tcount", "tmult"
        df = pandas.DataFrame(outputs, columns=cols)
        df_f = os.path.join(args.outdir, outname+ ".pkl")
        df.to_pickle(df_f)
        txt_f = os.path.join(args.outdir, outname+ ".tsv")
        df.to_csv(txt_f, sep="\t", float_format="%.3f", index=False)
        if args.makeMaxImgs:
            # max composite of top 10 images in res
            print("Max comps for res.")
            num_f = len(fnames)
            num_mx = int(num_f*0.05)
            d_mx_high, d_mx_low = get_mx_imgs(df, by="ai_d", num=num_mx)
            print("Max comps for mu.")
            mu_mx_high, mu_mx_low = get_mx_imgs(df, by="multi", num=num_mx)

            np_f = os.path.join( args.outdir, "%s_mx.npz" % outname)
            np.savez(np_f,
                     dhigh=d_mx_high, dlow=d_mx_low, muhigh=mu_mx_high, mulow=mu_mx_low)
    

if __name__ == "__main__":
    main()
