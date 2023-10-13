from argparse import ArgumentParser

parser = ArgumentParser()
from copy import deepcopy
parser.add_argument("file", type=str)
parser.add_argument("--out", type=str, default="test")
parser.add_argument("--mask", type=str, default=None)
parser.add_argument("--geom", type=str, default=None)
parser.add_argument("--reso", type=str, default=None)
parser.add_argument("--thresh", type=float, default=0)
parser.add_argument("--ndev", type=int, default=4)
parser.add_argument("--darkz", type=float, default=7)
parser.add_argument("--minSpotSize", type=int, default=None)
parser.add_argument("--kernelSize", default=None, nargs=2, type=int)
parser.add_argument("--dmin", default=None, type=float)
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

from mpi4py import MPI
COMM = MPI.COMM_WORLD
import os
import h5py
import numpy as np
import torch
from torch.nn import MaxPool2d
from torchvision import transforms

from resonet.utils import eval_model
from resonet.cxidb.FormatCheetah import FormatCheetah
from resonet.cxidb.FormatBigPixJungzy import FormatBigPixJungzy
from resonet.cxidb.FormatHDF5ImageDictionary import FormatHDF5ImageDictionary
from resonet.utils import multi_panel
from resonet.utils.predict import d_to_dnew

from dxtbx.model import Experiment, ExperimentList
from simtbx.nanoBragg import make_imageset
from dxtbx.model import ExperimentList, Experiment
from simtbx.diffBragg import utils
from dials.algorithms.spot_finding.per_image_analysis import estimate_resolution_limit
from dials.array_family import flex
from dials.command_line.stills_process import phil_scope
from cctbx import crystal
from dials.algorithms.indexing.indexer import Indexer
from dials.algorithms.indexing import DialsIndexError


def print0(*args, **kwargs):
    if COMM.rank==0:
        print(*args, **kwargs)

def clean_edge(img, s=2):
    for i in range(img.shape[0]):
        pan = img[i]
        pan[:s,:] = -1
        pan[:, :s] = -1
        pan[-s:, :] = -1
        pan[:, -s:] = -1
        img[i] = pan
    return img

ucells = {
    "r0018_2000.h5": (73,96,119,90,90,90),
    "r0095_2000.h5": (118,223,311,90,90,90),
    "r0020_2000.h5": (42,52,88,90,90,90),
    "cxidb164.loc": (79.1, 79.1, 38.2, 90, 90, 90),
    "cxidb17.loc": (78.66, 78.66, 37.75, 90,90, 90),
    "138k.loc": (78.66, 78.66, 37.75, 90,90, 90),
    "run795_dx.loc": (78.68, 78.68, 265.5, 90, 90, 120)
}

symbols={
    "r0018_2000.h5": "P212121",
    "r0095_2000.h5": "P212121",
    "r0020_2000.h5": "P212121",
    "cxidb164.loc": "P43212",
    "cxidb17.loc": "P43212",
    "138k.loc": "P43212",
    "run795_dx.loc": "P6522"
}


gains={
    "r0018_2000.h5": 0.31,
    "r0095_2000.h5": 0.27,
    "r0020_2000.h5": 0.19 ,
    "cxidb164.loc": 1.2659, # I THINK adu_per_eV * photon_energy
    "cxidb17.loc": 9.09,
    "138k.loc": 9.09 ,
    "run795_dx.loc": 9.5
}

formats={
    "r0018_2000.h5": FormatHDF5ImageDictionary, 
    "r0095_2000.h5": FormatHDF5ImageDictionary,  
    "r0020_2000.h5": FormatHDF5ImageDictionary,  
    "cxidb164.loc": FormatCheetah,
    "cxidb17.loc": FormatCheetah,
    "138k.loc": FormatCheetah,
    "run795_dx.loc": FormatBigPixJungzy
}


basename = os.path.basename(args.file)
SYMBOL = symbols[basename]
GAIN = gains[basename]
UCELL = ucells[basename]
MASK = None

sym = crystal.symmetry(UCELL, SYMBOL)

params = phil_scope.extract()

params.spotfinder.threshold.algorithm = "dispersion"
if args.minSpotSize is not None:
    params.spotfinder.filter.min_spot_size = args.minSpotSize
if args.kernelSize is not None:
    params.spotfinder.threshold.dispersion.kernel_size = args.kernelSize
if args.dmin is not None:
    params.spotfinder.filter.d_min = args.dmin
params.spotfinder.threshold.dispersion.gain = GAIN
params.spotfinder.threshold.dispersion.global_threshold = args.thresh
if args.mask is not None:
    params.spotfinder.lookup.mask = args.mask
    if basename=="run795_dx.loc":
        MASK = utils.load_mask(args.mask)
    else: 
        MASK = utils.load_mask(args.mask)[0]

params.indexing.known_symmetry.unit_cell=sym.unit_cell()
params.indexing.known_symmetry.space_group=sym.space_group_info()
params.indexing.method='fft1d'
params.indexing.stills.candidate_outlier_rejection=True
params.indexing.stills.refine_all_candidates=True
params.indexing.stills.refine_candidates_with_known_symmetry=True
params.indexing.multiple_lattice_search.max_lattices=1


FORMAT = formats[basename]
loader = FORMAT(args.file)

DET = loader.get_detector(0)
if args.geom is not None:
    DET = ExperimentList.from_file(args.geom, False)[0].detector
BEAM = loader.get_beam()
detdist = DET[0].get_distance() #93 
wavelen = BEAM.get_wavelength()
pixsize = DET[0].get_pixel_size()[0]
factor = 2
xdim, ydim = DET[0].get_image_size()
npan = len(DET)
if MASK is None:
    if basename=="run795_dx.loc":
        MASK = np.ones((len(DET), ydim, xdim), bool)
    else: 
        MASK = np.ones((ydim, xdim), bool)

if basename in ["r0095_2000.h5", "r0018_2000.h5", "r0020_2000.h5"]:
    h = h5py.File(args.file, "r")
    centx = h['metadata']['BEAM_CENTER_X'][()]
    centy = h['metadata']['BEAM_CENTER_Y'][()]
    centx, centy = int(centx/pixsize), int(centy/pixsize)
elif basename=="cxidb164.loc":
    centx, centy = DET[0].get_beam_centre_px(BEAM.get_unit_s0())
elif basename in ["cxidb17.loc", '138k.loc', 'run795_dx.loc']:
    E = Experiment()
    E.detector = DET
    E.beam = BEAM
    if basename=='run795_dx.loc':
        MASK_temp = MASK.copy()
    else:
        MASK_temp = np.array([panel.as_numpy_array() for panel in loader.get_raw_data(-1)] )
        MASK_temp = MASK_temp.astype(bool)

    mask, beam_center = multi_panel.project_jungfrau(E, 
        img=MASK_temp.astype(float), return_center=True)
    centx, centy = beam_center

else:
    centx = centy = None
    mask = None

# MODELS
reso_model = "/pscratch/sd/d/dermen/train.8_trial.1.restart/nety_ep700.nn"
multi_model = "/global/cfs/cdirs/lcls/dermen/d9114_sims/CrystalNew/modules/resonet/test.ddp.18/nety_ep54.nn"

if args.reso is not None:
    reso_model = args.reso

small_det = False
if xdim < 2048 or ydim < 2048:
    small_det = True

maxpool = MaxPool2d(kernel_size=factor, stride=factor)
geom = torch.tensor([[detdist, pixsize, wavelen, factor]])

dev_id = COMM.rank % args.ndev
dev = "cuda:%d" % dev_id
if args.cpu:
    dev = "cpu"
geom = geom.to(dev)

print0("loading models")
RESO = eval_model.load_model(reso_model, "res50").to(dev)
MULTI =  eval_model.load_model(multi_model, "res34").to(dev)

print0("format class")
Nimg = loader.get_num_images()

all_reso, all_multi = [], []

mode = transforms.InterpolationMode.BICUBIC
rs = transforms.Resize((2060,2060), interpolation=mode, antialias=True)

cent = None

if basename in ["r0095_2000.h5", "r0018_2000.h5", "r0020_2000.h5"]:
    old_ydim, old_xdim = MASK.shape
    mask = rs(torch.tensor(MASK[None]))[0].numpy()
    mask = mask.astype(bool)

    new_ydim, new_xdim = mask.shape
    new_centx = centx + (new_xdim - old_xdim)/2.
    new_centy = centy + (new_ydim - old_ydim)/2.
    cent = new_centx, new_centy

elif basename =="cxidb164.loc":
    cent = centx, centy
    mask = MASK.copy()

elif basename in ["cxidb17.loc", "138k.loc", "run795_dx.loc"]:
    old_ydim, old_xdim = mask.shape
    mask = rs(torch.tensor(mask[None]))[0].numpy()
    mask = mask.astype(bool)

    new_ydim, new_xdim = mask.shape
    new_centx = centx + (new_xdim - old_xdim)/2.
    new_centy = centy + (new_ydim - old_ydim)/2.
    cent = new_centx, new_centy


shot_data = []
for i_img in range(Nimg):
    if i_img % COMM.size != COMM.rank:
        continue
    try:
        data = loader.get_raw_data(i_img)
    except:
        continue
    if isinstance(data, tuple):
        if len(data)==1:
            img = data[0].as_numpy_array()
            iset_data = (img,)
        else:
            panels = np.array([panl.as_numpy_array() for panl in data])
            iset_data = tuple(panels)
            
            panels = clean_edge(panels)

            # map the panels to two-dims
            img = multi_panel.project_jungfrau(E, img=panels)

            # optionally shift the dark subtraction (since resampling takes the sqrt)
            OFFSET = 0
            if args.darkz is not None:
                vals = -1*img[img<0]
                vals = vals[~utils.is_outlier(vals, args.darkz)]
                OFFSET = vals.max()
            img = img + OFFSET

    else:
        img = data.as_numpy_array()
        iset_data = (img,)
        
    iset = make_imageset(iset_data, BEAM, DET)
    E = Experiment()
    E.detector = DET #det
    E.beam = BEAM 
    E.imageset = iset
    El = ExperimentList()
    El.append(E)
    R = flex.reflection_table.from_observations(El, params, is_stills=True)
    R.centroid_px_to_mm(El)
    R.map_centroids_to_reciprocal_space(El)

    Nref = len(R)
    dest = -1
    if Nref > 0:
        dest = estimate_resolution_limit(R)

    Nidx = Nlat = rmsd = 0
    try:
        idxr = Indexer.from_parameters(R, El, params=params)
        idxr.index()
        Ridx = idxr.refined_reflections
        Cs = idxr.refined_experiments.crystals()
        Nidx = len(Ridx)
        Nlat = len(Cs)
        x1,y1,_ = Ridx['xyzobs.px.value'].parts()
        x2,y2,_ = Ridx['xyzcal.px'].parts()
        dist = (x1-x2)**2 + (y1-y2)**2
        rmsd = np.sqrt(np.mean(dist))
    except (AssertionError, DialsIndexError, Exception) as err:
        pass
 
    if small_det:
        img = rs(torch.tensor(img[None]))[0].numpy()
    img = img.astype(np.float32)/GAIN
    quad_names = "A", "B", "C", "D"
    resos = []
    multis = []
    for i_quad, quad_name in enumerate(quad_names):
        quad = eval_model.to_tens(
            img, mask, quad=quad_name, dev=dev, cent=cent,
            maxpool=maxpool, ds_fact=factor)
        one_over_reso = RESO(quad, geom).item()
        reso = 1/one_over_reso
        reso = d_to_dnew(reso)
        is_multi = torch.sigmoid(MULTI(quad)).item()
        resos.append(reso)

        all_reso.append( (i_quad, i_img, reso))
        all_multi.append( (i_quad, i_img, is_multi))
        multis.append(is_multi)

    dmin,dave = np.min(resos), np.mean(resos)
    shot_data.append( (dmin,dave,i_img, dest, Nref, np.mean(multis), np.max(multis), Nidx, Nlat, rmsd) )
    print("Res Min,Ave,DIALS= (%.1f, %.1f, %.1f), Num ref= %d, Nidx=%d, Nlat=%d, Mult=%.1f, rmsd=%.2f (%d/%d)" 
        % (dmin,dave,dest, Nref, Nidx, Nlat, np.mean(multis), rmsd, i_img+1, Nimg))    

all_reso = COMM.reduce(all_reso)
all_multi = COMM.reduce(all_multi)
shot_data = COMM.reduce(shot_data)

if COMM.rank==0:
    np.savez(args.out , all_reso=all_reso, all_multi=all_multi, shot_data=shot_data)

