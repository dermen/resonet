from argparse import ArgumentParser
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
except ModuleNotFoundError:
    class nompi_comm:
        rank = 0
        size = 1
        def reduce(self, val):
            return val
    COMM = nompi_comm()

import Pyro4
import os
import glob
from resonet.utils import predict_dxtbx
import time
from numpy import median


@Pyro4.expose
class imageMonster:
    def __init__(self, P, kind):
        """P is an instance of predict_dxtbx"""
        self.P = P
        self.kind=kind

    @Pyro4.oneway
    def eat_images(self, glob_s, max_proc=None):
        # TODO:  add a Break button to break out of the loop using the mouse!
        seen = 0
        Nf = 0
        t_infers = []
        t_reads = []
        try:
            fnames = sorted(glob.glob(glob_s))
            Nf = len(fnames)
            if COMM.rank==0:
                print("Found %d shots in %s" % (Nf, glob_s), flush=True)
            for i_f, f in enumerate(fnames):
                if max_proc is not None and i_f >  max_proc:
                    break
                if i_f % COMM.size != COMM.rank:
                    continue
                t = time.time()
                try:
                    self.P.load_image_from_file(f)
                    t_reads.append( time.time()-t )
                except Exception as err:
                    print(err)
                    continue
                t=time.time()
                if self.kind=="reso":
                    d = self.P.detect_resolution()
                    t_infers.append( time.time() - t)
                    msg =  "Resolution estimate: %.3f Angstrom." % d
                else:
                    pval = self.P.detect_multilattice_scattering(binary=False)
                    t_infers.append(time.time()-t)
                    msg = "Chance that shot contains overlapping lattices: %.4f%%" % (pval*100)
                seen += 1
                print("Rank%d" % COMM.rank, os.path.basename(f), msg, "(%d/%d)"% (i_f+1, Nf), flush=seen % 10 == 0)
        except Exception as err:
            print(err, flush=True)
        total_seen = COMM.reduce(seen)
        total_reads = COMM.reduce(t_reads)
        total_infers = COMM.reduce(t_infers)
        if COMM.rank==0:
            t_read = -1
            t_infer = -1
            if total_reads:
                t_read = median(total_reads)*1e3
                t_infer = median(total_infers)*1e3
            print("Done. Processed %d / %d shots in total." % (total_seen, Nf), flush=True)
            print("Median inference time=%.4f milliseconds. Median time to load image from disk=%.4f milliseconds"
                  % (t_infer, t_read), flush=True)
        return None


def main():
    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="path to the .nn file storing the trained PyTorch model")
    parser.add_argument("arch", type=str, choices=["res50","res34","res18"], help="string specifying ResNet number, see arches.py")
    parser.add_argument("--kind", default="reso", type=str, choices=["reso", "multi"])
    parser.add_argument("--gpu", action="store_true", help="use GPU for inference")
    args = parser.parse_args()

    dev = "cpu"
    if args.gpu:
        dev="cuda:0"
    kwargs = {"dev": dev, "%s_model" % args.kind: args.model, "%s_arch" % args.kind: args.arch}
    print("Rank %d Initializing predictor" % COMM.rank, flush=True)
    P = predict_dxtbx.ImagePredictDxtbx(**kwargs)
    P.quads = [-2]  # uses two randomly chosen quads
    P.cache_raw_image = True
    dm = Pyro4.Daemon()
    name = Pyro4.locateNS()
    img_monst = imageMonster(P, args.kind)
    uri = dm.register(img_monst)
    name.register("image.monster%d" % COMM.rank, uri)
    print("Rank %d is ready to consume images... " % COMM.rank, uri, flush=True)
    dm.requestLoop()


if __name__=="__main__":
    main()
