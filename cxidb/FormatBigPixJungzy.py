import h5py
import numpy as np
from serialtbx.detector.jungfrau import pad_stacked_format
from dials.array_family import flex
from dxtbx.format.FormatStill import FormatStill
from dxtbx.model import Beam, Spectrum, ExperimentList
from simtbx.nanoBragg.utils import ENERGY_CONV

"""
This allows one to utilize the full Jungfrau 1030x514 pixel panel
"""


phil = """
files {
  raw = None
    .type = str
    .help = path to the raw data file
  dark = None
    .type = str
    .help = path to the pedestal file
  gain = None
    .type = str
    .help = path to the gain file
  beam = None
    .type = str
    .help = path to the beam file
  geom = None
    .type = str
    .help = path to the crystfel geometry (requires cfelpyutils installed via pip) 
    .help = or path to the dxtbx geometry (experimentList file)
}
paths {
  raw_data = 'data/JF07T32V01/data'
    .type = str
    .help = path of the raw dataset in the files.raw 
  raw_pulses = 'data/JF07T32V01/pulse_id'
    .type = str
  dark = 'gains'
    .type = str
    .help = path of the gain dataset in files.dark
  gain = 'gains'
    .help = path of the raw dataset in the files.gain 
    .type = str
  beam_pulses = 'data/SARFE10-PSSS059:SPECTRUM_CENTER/pulse_id'
    .type = str
}
geom_type = *crystfel dxtbx
  .type = choice
  .help = type of the files.geom
"""


from libtbx.phil import parse

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

from dxtbx.model import Detector, Panel
from copy import deepcopy


class cfelDetector:
    def __init__(self, geomfile):
        
        try:
            from cfelpyutils import geometry
        except ImportError:
            raise ImportError("To use the crystfel geometry, please install cfelpyutils using 'pip install cfelpyutils'")
        
        G = geometry.load_crystfel_geometry(geomfile)
        det = G[0]['panels']
        self.num_panels = len(det)
        self.panels = [cfelPanel(det[m]) for m in det]

    def __getitem__(self, item):
        return self.panels[item]

    def to_dxtbx(self):
        DET = Detector()
        for pid, pan in enumerate(self):
            pd = deepcopy(PDICT)
            ox,oy,oz = pan.origin
            fx,fy,fz = pan.fast_axis
            sx,sy,sz = pan.slow_axis
            pd["origin"] = -ox,-oy,oz
            pd["fast_axis"] = -fx,-fy,fz
            pd["slow_axis"] = -sx,-sy,sz
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
        self.pixsize_mm = 1./self.res * 1e3
        self.fast_axis = self.fsx, self.fsy, self.fsz
        self.slow_axis = self.ssx, self.ssy, self.ssz
        self.cnz = self.clen + self.coffset
        self.origin = self.cnx*self.pixsize_mm, self.cny*self.pixsize_mm, self.cnz*1e3


beam_descr = {'direction': (0.0, 0.0, 1.0),
             'divergence': 0.0,
             'flux': 1e12,
             'polarization_fraction': 1.,
             'polarization_normal': (0.0, 1.0, 0.0),
             'sigma_divergence': 0.0,
             'transmission': 1.0,
             'wavelength': 1.4}

MASTER_PHIL = parse(phil)

from dxtbx.format.Format import Format
from dxtbx.format.FormatMultiImageLazy import FormatMultiImageLazy


class FormatBigPixJungzy(FormatMultiImageLazy, FormatStill, Format):
    def __init__(self, image_file, **kwargs):
        FormatMultiImageLazy.__init__(self, **kwargs)
        FormatStill.__init__(self, image_file, **kwargs)
        Format.__init__(self, image_file, **kwargs)

    @staticmethod
    def understand(image_file, quiet=True):
        try:
            user_phil = parse(open(image_file, "r").read())
            params = MASTER_PHIL.fetch(sources=[user_phil]).extract()
            nfail = 0
            nfail += int(params.files.raw is None)
            nfail += int(params.files.dark is None)
            nfail += int(params.files.gain is None)
            nfail += int(params.files.geom is None)
            if nfail > 0:
                if not quiet:
                    print("Fails!")
                return False
            else:
                return True
        except Exception as err:
            if not quiet:
                print("Fails!", err)
            return False

    def _start(self):
        super(FormatBigPixJungzy, self)._start()

        user_phil = parse(open(self._image_file, "r").read())
        self.params = MASTER_PHIL.fetch(sources=[user_phil]).extract()

        raw = h5py.File(self.params.files.raw, "r")
        dark = h5py.File(self.params.files.dark, "r")
        gain = h5py.File(self.params.files.gain, "r")
        beam = h5py.File(self.params.files.beam, 'r')

        # make the detector
        if self.params.geom_type=="crysfel":
            geom = cfelDetector(self.params.files.geom)
            self._dxtbx_detector = geom.to_dxtbx()
            self.beam_descr = deepcopy(beam_descr)
            self.beam_descr["direction"] = 0, 0, -1
        else:
            geom = ExperimentList.from_file(self.params.files.geom, False)
            self._dxtbx_detector = geom[0].detector
            self.beam_descr = deepcopy(beam_descr)
            self.beam_descr["direction"] = 0, 0, 1

        self.RAW = raw[self.params.paths.raw_data]
        self.DARK = dark[self.params.paths.dark]
        self.DARK_RMS = dark[self.params.paths.dark+"RMS"]
        self.GAIN = gain[self.params.paths.gain]

        raw_pulses = raw[self.params.paths.raw_pulses][()]
        beam_pulses = beam[self.params.paths.beam_pulses][()]
        self.beam_energy = beam['data/SARFE10-PSSS059:SPECTRUM_X/data'][()]
        self.beam_weights = beam['data/SARFE10-PSSS059:SPECTRUM_Y/data'][()]

        assert np.allclose(np.sort(beam_pulses), beam_pulses)
        self._spectra_idx = np.searchsorted(beam_pulses, raw_pulses)[:,0]

        A = np.sum(self.beam_energy*self.beam_weights, axis=1)
        B = np.sum(self.beam_weights, axis=1)
        self._nominal_energies = np.array([a/b if b >0 else np.nan for a,b in zip(A,B)])

    def get_beam(self, index=None):
        beam_model = Beam.from_dict(self.beam_descr)
        if index is not None:
            nom_en = self._nominal_energies[self._spectra_idx[index]]
            beam_model.set_wavelength(ENERGY_CONV /nom_en)
        return beam_model

    def get_num_images(self):
        return self.RAW.shape[0]

    def get_detectorbase(self, index=None):
        raise NotImplementedError

    def get_pedestal_rms(self, index=None, return_gain_modes=False):
        shape = 16384, 1024
        img = np.zeros(shape)

        raw = self.RAW[index]
        raw_gain = raw >> 14  # 1100000000000000  , gain are the first 2 bits
        for mode in (0, 1, 3):
            mode_idx = mode if mode != 3 else 2
            sel = raw_gain == mode
            img[sel] = self.DARK_RMS[mode_idx][sel] / self.GAIN[mode_idx][sel]

        corrected = pad_stacked_format(img, num_panels=32, divide=True, keep_stacked=False)
        dark_rms = np.array(corrected)

        return dark_rms

    def get_pedestal_rms_for_gain_mode(self, mode=0):
        mode_idx = mode if mode != 3 else 2
        img = self.DARK_RMS[mode_idx] / self.GAIN[mode_idx]
        corrected = pad_stacked_format(img, num_panels=32, divide=True, keep_stacked=False)
        dark_rms = np.array(corrected)

        return dark_rms

    def get_raw_data(self, index):
        shape = 16384, 1024
        img = np.zeros(shape)

        raw = self.RAW[index]
        raw_gain = raw >> 14  # 1100000000000000  , gain are the first 2 bits
        raw_counts = np.array(raw & 0x3FFF).astype(np.float64)  # 0011111111111111  , counts are the last 14 bits
        for mode in (0, 1, 3):
            mode_idx = mode if mode != 3 else 2
            sel = raw_gain == mode
            img[sel] = (raw_counts[sel] - self.DARK[mode_idx][sel]) / self.GAIN[mode_idx][sel]

        corrected = pad_stacked_format(img, num_panels=32, divide=True, keep_stacked=False)
        imgset_data = tuple([flex.double(p) for p in corrected])

        return imgset_data

    def get_detector(self, index=None):
        return self._dxtbx_detector

    def get_spectrum(self, index=0):
        en = self.beam_energy[self._spectra_idx[index]]
        wt = self.beam_weights[self._spectra_idx[index]]
        en = np.ascontiguousarray(en.astype(np.float64))
        wt = np.ascontiguousarray(wt.astype(np.float64))
        spectrum = Spectrum(en, wt)
        return spectrum


if __name__ == '__main__':
    """
    This creates a mask for the BigPixJungzy format for Swissfel Berninina data collected in Oct 2018
    """
    import sys
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("imgfile", type=str)
    parser.add_argument("--thresh", type=float, default=5, help="MAD threshhold for mask-maker")
    parser.add_argument("--outfile", type=str, default="bigPixJungzy.mask")
    parser.add_argument("--border", type=int, default=3, help="border mask")
    parser.add_argument("--plot", action="store_true")
    #parser.add_argument("--numshots", type=int, default=20, help="number of shots for median")
    args = parser.parse_args()
    from simtbx.diffBragg import utils
    assert FormatBigPixJungzy.understand(args.imgfile, quiet=False)
    Fclass = FormatBigPixJungzy(args.imgfile)
    #shots = []
    #for i_shot in range(args.numshots):
    #    shot = Fclass.get_raw_data(i_shot)
    #    shot = np.array([d.as_numpy_array() for d in shot])
    #    shots.append(shot)
    #    print(i_shot)
    #shots = np.median(shots,0)

    # use the pedestal RMS from each gain mode to automatically mask bad pixels
    d0 = Fclass.get_pedestal_rms_for_gain_mode(0)
    d1 = Fclass.get_pedestal_rms_for_gain_mode(1)
    d3 = Fclass.get_pedestal_rms_for_gain_mode(3)
    pan_sh = d0[0].shape
    npanel = len(d0)

    # main mask
    mask = np.ones((npanel,)+pan_sh).astype(np.bool)
    
    # mask the border
    mask[:, :args.border, :] = False
    mask[:, :, :args.border] = False
    mask[:, -args.border:, :] = False
    mask[:, :, -args.border:] = False
    
    # swissFEL touch ups from visial inspection of the median (see `shots` variable above)
    mask[0, :, 256:517] = False
    mask[5] = False
    mask[13, :, :260] = False
    mask[13, 500:, 960:] = False
    mask[14, 460:, :20] = False
    mask[14, 72:255, 228:231] = False
    mask[27,116:258,598:601] = False
    mask[27,192:195,580:610] = False
    mask[27, 275:, 615:] = False
    mask[28,252:,:520] = False
    mask[28,185,356] = False
    mask[29,:256,500:] = False
    mask[30,256:468,347:350] = False
    mask[31,:,600:] = False
    mask[31,59,413] = False
    mask[31,76,425:427] = False
    mask[31,23:25,513:515] = False
    mask[31,123:126,372:515] = False
    mask[31,116:120,500:514] = False
    mask[21,250:259,:17] = False
    mask[27,108:260,570:750] = False
    mask[30,249:480,256:420] = False

    # for each gain mode, mask faulty pixels based on a simple outlier detection algorithm
    for d in d0, d1, d3:
        print(1)
        for pid in range(npanel):
            dp = d[pid].ravel()
            mp = mask[pid].ravel()
            pix_idx = np.arange(len(dp))[mp]
            out = utils.is_outlier(dp[mp], args.thresh)
            dp[pix_idx[out]] = False
            mask[pid] = np.logical_and(mask[pid], dp.reshape(pan_sh))
    nmasked= np.sum(mask)

    # optionally display the masked panels
    if args.plot:
        import pylab as plt
        for pid in range(npanel):
            plt.gca().clear()
            plt.imshow(mask[pid])
            plt.title(str(pid))
            plt.draw()
            plt.pause(1.2)

    frac_masked = nmasked / float(mask.size)
    print("Masked %f %% of pixels" % frac_masked)

    utils.save_numpy_mask_as_flex(mask, args.outfile)
    print("Wrote mask to %s." % args.outfile)
