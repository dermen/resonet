from dxtbx.model import Detector, Panel
from dxtbx.format.FormatStill import FormatStill
from dxtbx.format.Format import Format
from dxtbx.format.FormatMultiImageLazy import FormatMultiImageLazy
from copy import deepcopy
from cfelpyutils import geometry
from dials.array_family import flex
from dxtbx.model import BeamFactory
from libtbx.phil import parse
import h5py
import numpy as np

phil = """
files {
  geom = None
    .type = str
    .help = path to the raw data file
  hits = None
    .type = str
    .help = path to text file pointing to cheetah hits
  mask = None
    .type = str
}
wavelength = None
  .type = float
  .help = wavelength of photons in Angstrom
"""

MASTER_PHIL = parse(phil)
PDICT = {'fast_axis': None,
         'gain': 1.0,
         'identifier': '',
         'image_size': None,
         'mask': [],
         'material': '',
         'mu': 0.0,
         'name': 'Panel',
         'origin': None,
         'pedestal': 0.0,
         'pixel_size': None,
         'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
         'slow_axis': None,
         'thickness': 0.0,
         'trusted_range': (-100,255000),
         'type': ''}


class cfelDetector:
    def __init__(self, geomfile):

        try:
            from cfelpyutils import geometry
        except ImportError:
            raise ImportError(
                "To use the crystfel geometry, please install cfelpyutils using 'pip install cfelpyutils'")

        self.G = geometry.load_crystfel_geometry(geomfile)
        det = self.G[0]['panels']
        self.num_panels = len(det)
        self.panels = [cfelPanel(det[m]) for m in det]

    def __len__(self):
        return len(self.panels)

    def __getitem__(self, item):
        return self.panels[item]

    def to_dxtbx(self):
        DET = Detector()
        for pid, pan in enumerate(self):
            pd = deepcopy(PDICT)
            ox, oy, oz = pan.origin
            fx, fy, fz = pan.fast_axis
            sx, sy, sz = pan.slow_axis
            pd["origin"] = -ox, -oy, oz
            pd["fast_axis"] = -fx, -fy, fz
            pd["slow_axis"] = -sx, -sy, sz
            pd["pixel_size"] = pan.pixsize_mm, pan.pixsize_mm
            pd["name"] = "M%d" % pid
            nfast = pan.orig_max_fs - pan.orig_min_fs + 1
            nslow = pan.orig_max_ss - pan.orig_min_ss + 1
            pd['image_size'] = nfast, nslow
            dxtbx_pan = Panel.from_dict(pd)
            DET.add_panel(dxtbx_pan)
        return DET


class cfelPanel:
    def __init__(self, panel_dict):
        for k in panel_dict:
            self.__setattr__(k, panel_dict[k])
        self.pixsize_mm = 1. / self.res * 1e3
        self.fast_axis = self.fsx, self.fsy, self.fsz
        self.slow_axis = self.ssx, self.ssy, self.ssz
        self.cnz = self.clen + self.coffset
        self.origin = self.cnx * self.pixsize_mm, self.cny * self.pixsize_mm, self.cnz * 1e3
        self.x1 = self.orig_min_fs
        self.y1 = self.orig_min_ss
        self.x2 = self.orig_max_fs + 1
        self.y2 = self.orig_max_ss + 1


class FormatCheetah(FormatMultiImageLazy, FormatStill, Format):
    def __init__(self, image_file, **kwargs):
        FormatMultiImageLazy.__init__(self, **kwargs)
        FormatStill.__init__(self, image_file, **kwargs)
        Format.__init__(self, image_file, **kwargs)

    @staticmethod
    def understand(image_file, quiet=True):
        try:
            user_phil = parse(open(image_file, "r").read())
            params = MASTER_PHIL.fetch(sources=[user_phil]).extract()
            assert params.files.geom is not None
            assert params.files.hits is not None
            assert params.wavelength is not None
            return True
        except Exception as err:
            return False

    def _start(self):
        super(FormatCheetah, self)._start()

        user_phil = parse(open(self._image_file, "r").read())
        self.params = MASTER_PHIL.fetch(sources=[user_phil]).extract()

        self.mask_raw = None
        if self.params.files.mask is not None:
            mask_handle = h5py.File(self.params.files.mask, "r")
            self.mask_raw = mask_handle["/data/data"][()].astype(np.float64)

        self.hits = [l.strip() for l in open(self.params.files.hits, "r").readlines()]

        # make the detector
        self.geom = cfelDetector(self.params.files.geom)
        self.npanel = len(self.geom)
        self._dxtbx_detector = self.geom.to_dxtbx()

        #self.RAW = raw["/data/data"]
        #if len(self.RAW.shape)==2:
        #    self.RAW = self.RAW[()][None]  # make it 1xSdimxFdim
        self.wave = self.params.wavelength
        self.beam_descr = {'direction': (0.0, 0.0, -1.0),
                      'divergence': 0.0,
                      'flux': 1e12,
                      'polarization_fraction': 1.,
                      'polarization_normal': (0.0, 1.0, 0.0),
                      'sigma_divergence': 0.0,
                      'transmission': 1.0,
                      'wavelength': self.wave}
        self._dxtbx_beam = BeamFactory.from_dict(self.beam_descr)
        # TODO:
        #self.wave = # get wavelength from geom file

    def get_beam(self, index=None):
        return self._dxtbx_beam

    def get_num_images(self):
        return len(self.hits)

    def get_detectorbase(self, index=None):
        raise NotImplementedError

    def get_raw_data(self, index):
        """

        :param index: assuming files.mask was provided, use index=-1 to get the mask!
        :return:
        """
        if index==-1:
            raw_frame = self.mask_raw
        else:
            hit_name = self.hits[index]
            raw_frame = h5py.File(hit_name, "r")["data/data"][()]

        panels = []
        for i_pan in range(self.npanel):
            # TODO get panel slice from geom file
            pan = self.geom[i_pan]
            raw_pan = raw_frame[pan.y1: pan.y2, pan.x1: pan.x2]
            if not raw_pan.dtype==np.float64:
                raw_pan = raw_pan.astype(np.float64)
            if not raw_pan.flags.c_contiguous:
                raw_pan = np.ascontiguousarray(raw_pan)
            panels.append(flex.double(raw_pan))
        return tuple(panels)

    def get_detector(self, index=None):
        return self._dxtbx_detector


if __name__=="__main__":
    import sys

    image_file = sys.argv[1]
    if FormatCheetah.understand(image_file):
        F = FormatCheetah(image_file)
        img = F.get_raw_data(0)
        print("Ok!")
