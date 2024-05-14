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
import time
from resonet.utils.predict import ImagePredict
import dxtbx
import numpy as np

import binascii
import h5py
import nxmx
from tqdm import tqdm
import dxtbx.nexus
import hdf5plugin


@Pyro4.expose
class imageMonster:
    def __init__(self, dev, model, kind, arch):
        """P is an instance of predict_dxtbx"""
        self.dev = dev
        self.model = model
        self.kind=kind
        self.arch  = arch

    def load_image_from_file(self, image_file, loader):
        """
        :param image_file:  path to an image file readable by DXTBX
        """
        try:
            raw_image = loader.get_raw_data()
            l = 0
            yield raw_image, l
        except:  # TODO put proper exception here
            with h5py.File(image_file, swmr=True) as f:
                nxmx_obj = nxmx.NXmx(f)
                nxsample = nxmx_obj.entries[0].samples[0]
                nxinstrument = nxmx_obj.entries[0].instruments[0]
                nxdetector = nxinstrument.detectors[0]
                nxdata = nxmx_obj.entries[0].data[0]
                dependency_chain = nxmx.get_dependency_chain(nxsample.depends_on)
                scan_axis = None
                for t in dependency_chain:
                    # Find the first varying rotation axis
                    if (
                        t.transformation_type == "rotation"
                        and len(t) > 1
                        and not np.all(t[()] == t[0])
                    ):
                        scan_axis = t
                        break
                if scan_axis is None:
                    # Fall back on the first varying axis of any type
                    for t in dependency_chain:
                        if len(t) > 1 and not np.all(t[()] == t[0]):
                            scan_axis = t
                            break
                if scan_axis is None:
                    scan_axis = nxsample.depends_on
                num_images = len(scan_axis)
                l = 0
                for k, j in enumerate(tqdm(range(num_images), unit=" images")):
                    l += 1
                    if k % COMM.size != COMM.rank:
                        continue
                    (raw_image,) = dxtbx.nexus.get_raw_data(nxdata, nxdetector, j)
                    yield raw_image, l

    

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
            image_predict = ImagePredict(dev=self.dev, 
                                         reso_model=self.model, 
                                         reso_arch=self.arch)
            image_predict.quads = [-2]  # uses two randomly chosen quads,
            image_predict.cache_raw_image = False
            m = 0
            for i_f, f in enumerate(fnames):
                m += 1
                if f.lower().endswith(".cbf"):
                    ftype = 'cbf'
                elif f.lower().endswith(".h5"):
                    ftype = 'h5'
                else:
                    ftype = 'unknown'
                if ftype == 'cbf':
                    if i_f % COMM.size != COMM.rank:
                        continue
                if max_proc is not None and i_f >=  max_proc:
                    break
                loader = dxtbx.load(f)
                det = loader.get_detector()
                beam = loader.get_beam()
                image_predict.xdim, image_predict.ydim = det[0].get_image_size()
                image_predict.pixsize_mm = det[0].get_pixel_size()[0]
                image_predict.detdist_mm = abs(det[0].get_distance())
                image_predict.wavelen_Angstrom = beam.get_wavelength()
                image_predict._set_geom_tensor()
                if len(det) > 1:
                    raise NotImplementedError("Not currently supporting multi panel formats")
                t=time.time()
                i = 0
                imgs = 0
                for raw_image, l in self.load_image_from_file(f, loader):
                    #TODO: create singe ImagePredict object to get resolution

                    if max_proc is not None and i >=  max_proc:
                        break

                    t_reads.append( time.time()-t )
                    if isinstance(raw_image, tuple):
                        raw_image = np.array([panel.as_numpy_array() for panel in raw_image])
                    else:
                        raw_image = raw_image.as_numpy_array()
                    if not raw_image.dtype == np.float32:
                        raw_image = raw_image.astype(np.float32)
                    image_predict._set_pixel_tensor(raw_image)
                    if self.kind=="reso":
                        d = image_predict.detect_resolution()
                        t_infers.append( time.time() - t)
                        msg =  "Resolution estimate: %.3f Angstrom." % d
                    else:
                        pval = image_predict.detect_multilattice_scattering(binary=False)
                        t_infers.append(time.time()-t)
                        msg = "Chance that shot contains overlapping lattices: %.4f%%" % (pval*100)
                    seen += 1
                    if ftype == 'cbf':
                        resno = m
                    elif ftype == 'h5':
                        resno = l
                    print("\n Rank%d" % COMM.rank, os.path.basename(f), resno, msg, "(%d/%d)"% (i_f+1, Nf), flush=seen % 10 == 0)
                    i += 1
        except Exception as err:
            print(err, flush=True)
        total_seen = COMM.reduce(seen)
        total_reads = COMM.reduce(t_reads)
        total_infers = COMM.reduce(t_infers)
        if COMM.rank==0:
            t_read = -1
            t_infer = -1
            if total_reads:
                t_read = np.median(total_reads)*1e3
                t_infer = np.median(total_infers)*1e3
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
    dm = Pyro4.Daemon()
    name = Pyro4.locateNS()
    img_monst = imageMonster(dev, args.model, args.kind, args.arch)
    uri = dm.register(img_monst)
    name.register("image.monster%d" % COMM.rank, uri)
    print("Rank %d is ready to consume images... " % COMM.rank, uri, flush=True)
    dm.requestLoop()


if __name__=="__main__":
    main()
