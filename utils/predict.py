
import torch
from scipy.ndimage import binary_dilation
import numpy as np
from collections.abc import Iterable

from resonet.utils.eval_model import load_model, to_tens
from resonet.utils.ice_mask import IceMasker
from resonet.utils.counter_utils import mx_gamma, load_count_model, process_image

"""
"""


def d_to_dnew(d):
    """

    Parameters
    ----------
    d: resolution estimate from CRAIT assuming Holton 2009 model B = 4d^2 + 12

    Returns
    -------
    new resolution estimate, based on a modern fit to the PDB

    """
    B = 4 * d ** 2 + 12  # Holton 2009 model
    # quadratic fit coef for 2023 trend  (B = 15d^2 - 33d + 37)
    a, b, c = 15., -33., 37. - B
    sqrt_arg = b ** 2 - 4 * a * c
    if sqrt_arg < 0:  # super high res case
        # TODO: discuss whether its better to fall back on simply d (input to function) here
        # fall back on a linear fit to high-res data in the PDB
        # B = 18d - 4
        return d
        dnew = np.sqrt((B - 4) / 18)
        return dnew
    else:
        dnew = .5 * (-b + np.sqrt(sqrt_arg)) / a  # positive root
        return dnew


from resonet.utils.mlp_fit import CurveFitMLP


class ImagePredict:

    def __init__(self, reso_model=None, multi_model=None, ice_model=None, counts_model=None,
                 reso_arch=None, multi_arch=None, ice_arch=None, counts_arch=None,
                 dev="cpu", use_modern_reso=True, B_to_d=None):
        """

        Parameters
        ----------
        reso_model: resolution model path
        multi_model: splitting model path
        ice_model: ice detection model path
        counts_model: spot count model path
        reso_arch: resolution arch (e.g. res50)
        multi_arch: splitting arch
        ice_arch: ice detection arch
        counts_arch: spot count model arch
        dev: device string (e.g. 'cpu' or 'cuda:0')
        use_modern_reso: bool, whether to use the d_to_dnew method to alter resolution
        B_to_d: str, path to the MLP model for estimating reso from B factor
        """
        self.ice_masker = None   # instance of resonet.utiuls.ice_masker.IceMasker
        self.pixels = None  # this is the image tensor, a (512x512) representation of the diffraction shot
        self.counts_pixels = None  # downsampled tensor representation of the entire image
        self.geom = None  # the geometry tensor, (1x5) tensor with elements (detdist, wavelen, pixsize, xdim, ydim)
        self._dev = dev
        self.use_modern_reso = use_modern_reso
        self._try_load_model("reso", reso_model, reso_arch, load_model)
        self._try_load_model("multi", multi_model, multi_arch, load_model)
        self._try_load_model("ice", ice_model, ice_arch, load_model)
        self._try_load_model("counts", counts_model, counts_arch, load_count_model)
        self._try_load_B_to_d(B_to_d)

        self._geom_props = ["detdist_mm", "pixsize_mm", "wavelen_Angstrom", "xdim", "ydim"]
        self.mask = None  # True if pixel is valid
        self.ice_mask = None  # True if pixel is not ice

        self.maxpool_1x1 = torch.nn.MaxPool2d(1, 1)
        self.maxpool_2x2 = torch.nn.MaxPool2d(2, 2)
        self.maxpool_3x3 = torch.nn.MaxPool2d(3, 3)
        self.maxpool_4x4 = torch.nn.MaxPool2d(4, 4)
        self.maxpool_pilatus_counts = mx_gamma(self._dev, stride=3)
        self.maxpool_eiger_counts = mx_gamma(self._dev, stride=5)
        self.allowed_quads = {-1: "rand1", -2: "rand2", 0: "A", 1: "B", 2: "C", 3: "D"}
        self.quads = [1]
        self.ds_stride = None
        self.cent = None
        self.gain = 1  # adu per photon
        self.raw_image = None
        self.cache_raw_image = False

    def _try_load_B_to_d(self, path):
        """path: saved MLP model for estimating reso from Bfactor"""
        self.B_to_d_model = None
        if path is not None:
            self.B_to_d_model = CurveFitMLP.load_model(path)

    def _try_load_model(self, model_name, model_path, model_arch, method):
        """
        If model_path is None, model will have a value of None and prediction will be disabled.
        Otherwise, the model will be set.
        :param model_name: class attribute name of the model
        :param model_path: path to the model state file (.nn)
        :param model_arch: name of the model arch (see resonet.parameters or resonet.arches)
        :param method: method for loading model
        :return:
        """
        model = None
        if model_path is not None:
            if model_arch is None:
                raise ValueError("Arch string required for loading model %s!" % model_name)
            model = method(model_path, model_arch)
            model = model.to(self._dev)
        setattr(self, "%s_model" % model_name, model)

    @property
    def cache_raw_image(self):
        return self._cache_raw_image

    @cache_raw_image.setter
    def cache_raw_image(self, val):
        self._cache_raw_image = val

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, val):
        self._pixels = val

    @property
    def counts_pixels(self):
        return self._counts_pixels

    @counts_pixels.setter
    def counts_pixels(self, val):
        self._counts_pixels = val

    @property
    def geom(self):
        return self._geom

    @geom.setter
    def geom(self, val):
        self._geom = val

    @property
    def detdist_mm(self):
        return self._detdist_mm

    @detdist_mm.setter
    def detdist_mm(self, val):
        self._detdist_mm = val

    @property
    def pixsize_mm(self):
        return self._pixsize_mm

    @pixsize_mm.setter
    def pixsize_mm(self, val):
        self._pixsize_mm = val

    @property
    def wavelen_Angstrom(self):
        return self._wavelen_Angstrom

    @wavelen_Angstrom.setter
    def wavelen_Angstrom(self, val):
        self._wavelen_Angstrom = val

    @property
    def xdim(self):
        return self._xdim

    @xdim.setter
    def xdim(self, val):
        self._xdim = val

    @property
    def ydim(self):
        return self._ydim

    @ydim.setter
    def ydim(self, val):
        self._ydim = val

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, val):
        assert val > 0
        self._gain = val
    
    @property
    def cent(self):
        return self._cent

    @cent.setter
    def cent(self, val):
        if val is not None:
            assert isinstance(val, Iterable)
            assert len(val)==2
            assert isinstance(val[0], float) or isinstance(val[0], int)
            assert isinstance(val[1], float) or isinstance(val[1], int)

        self._cent = val

    @property
    def ds_stride(self):
        return self._ds_stride

    @ds_stride.setter
    def ds_stride(self, val):
        if val not in [1,2,3,4,None]:
            raise ValueError("Down sample stride should be 1-4 or None")
        self._ds_stride = val

    @property
    def quads(self):
        return self._quads

    @quads.setter
    def quads(self, val):
        # TODO assert val is iterable
        for v in val:
            if v not in self.allowed_quads:
                raise ValueError("Only values -1,-2,0,1,2,3 can be set as quads")
        if -1 in val or -2 in val:
            if not len(val)==1:
                raise ValueError("If -1 or -2 in quads, no other values are allowed")
        self._quads = [self.allowed_quads[v] for v in val]

    def _set_geom_tensor(self):
        for prop in self._geom_props:
            if getattr(self, prop) is None:
                raise ValueError("Must set %s before initializing geom tensor" % prop)

        if self.ds_stride is not None:
            self.geom = torch.tensor([[self.detdist_mm, self.pixsize_mm, self.wavelen_Angstrom, self.ds_stride]])
        else:
            self.geom = torch.tensor([[self.detdist_mm, self.pixsize_mm, self.wavelen_Angstrom, self.xdim, self.ydim]])
        self.geom = self.geom.to(self._dev)

    def set_ice_mask(self, dxtbx_geom=None, simple_geom=None):
        if self.ice_masker is None:
            self.ice_masker = IceMasker(dxtbx_geom, simple_geom)
        if simple_geom is not None:
            kwargs = {"distance": simple_geom["distance_mm"], "wavelength": simple_geom["wavelength_Ang"],
                      "beam_x": simple_geom["beam_x"], "beam_y": simple_geom["beam_y"]}
        else:
            assert dxtbx_geom is not None
            det = dxtbx_geom['detector']
            beam = dxtbx_geom["beam"]
            kwargs = {"distance": det[0].get_distance(), "wavelength": beam.get_wavelength()}
            beam_x, beam_y = det[0].get_beam_centre_px(beam.get_unit_s0())
            kwargs["beam_x"] = beam_x
            kwargs["beam_y"] = beam_y

        self.ice_mask = ~self.ice_masker.mask(**kwargs)[0]

    def _set_pixel_tensor(self, raw_img):
        """pass in a raw image (2D array) and convert it to an torch tensor for prediction"""
        # check for mask and set a default if none found
        self._set_default_mask(raw_img)

        dwnsamp = self.ds_stride
        if dwnsamp is None:
            is_pil = self.xdim == 2463
            print('\n is pil', is_pil)
            if is_pil:
                dwnsamp = 2
                maxpool = self.maxpool_2x2
            else:  # eiger or some other large format
                maxpool = self.maxpool_4x4
                dwnsamp = 4
        maxpool = getattr(self, "maxpool_%dx%d" % (dwnsamp, dwnsamp))
        tensors = []
        _quads = self.quads
        if _quads == ['rand1'] or _quads == ['rand2']:
            size = 1 if _quads== ['rand1'] else 2
            _quads = np.random.choice(["A", "B", "C", "D"], size=size, replace=False)
        for quad in _quads:
            tens = to_tens(raw_img/self.gain, self.mask, cent=self.cent, maxpool=maxpool,
                           ds_fact=dwnsamp, quad=quad, dev=self._dev)
            tensors.append(tens)
        self.pixels = torch.concatenate(tensors)

        if self.counts_model is not None:
            mx = self.maxpool_pilatus_counts
            if not is_pil:
                mx = self.maxpool_eiger_counts
            self.counts_pixels = process_image(raw_img/self.gain*self.mask, cond_meth=mx,
                                       useSqrt=True, dev=self._dev)[None]

        if self.cache_raw_image:
            self.raw_image = raw_img

    def _set_default_mask(self, raw_img):
        if self.mask is None or raw_img.shape != self.mask.shape:
            mask = raw_img >= 0
            mask = ~binary_dilation(~mask, iterations=1)
            self.mask = mask
        if self.ice_mask is not None:
            assert raw_img.shape == self.ice_mask.shape
            self.mask = np.logical_and(self.mask, self.ice_mask)

    def detect_resolution(self, use_min=True):
        """
        :return: an estimate of the crystal resolution in Angstroms
        """
        self._check_pixels()
        self._check_geom()
        self._check_model("reso")

        one_over_reso = self.reso_model(self.pixels, self.geom)

        if use_min:
            reso = torch.min(1/one_over_reso).item()
        else:
            reso = torch.mean(1/one_over_reso).item()

        if self.B_to_d_model is not None:
            reso = self.d_from_MLP(reso)
        elif self.use_modern_reso:
            reso = d_to_dnew(reso)
        return reso

    def d_from_MLP(self, d):
        B = 4 * d ** 2 + 12  # Holton 2009 model
        B = torch.tensor([[B]])
        d = self.B_to_d_model(B).item()
        return d

    def count_spots(self):
        """
        :return: an estimate of the number of spot in the image
        """
        self._check_counts_pixels()
        #self._check_geom()
        self._check_model("counts")
        counts = self.counts_model(self.counts_pixels)
        counts = counts.item()
        return counts

    def detect_multilattice_scattering(self, binary=True):
        """
        :param binary: whether to return a binary number, or a float between 0 and 1
        :return: 1 (multilattice scattering detector) or 0 (single lattice scattering detected), or else a floating
                value between 0 and 1
        """
        self._check_pixels()
        self._check_model("multi")
        self.multi_model(self.pixels)
        raw_prediction = self.multi_model(self.pixels)
        raw_prediction = torch.sigmoid(raw_prediction)
        raw_prediction = torch.mean(raw_prediction)
        if binary:
            is_multi = int(torch.round(raw_prediction).item())
            return is_multi
        else:
            return raw_prediction.item()

    def detect_ice(self):
        self._check_pixels()
        self._check_model("ice")
        raise NotImplementedError("Still training models!")

    def _check_model(self, model_name):
        attr_name = "%s_model" % model_name
        model = getattr(self, attr_name)
        if model is None:
            raise ValueError("{model_name}_model path was not provided when instantiating this class. "
                             "Cant detect `{model_name}` scattering!".format(model_name=model_name))

    def _check_pixels(self):
        if self.pixels is None:
            raise ValueError("pixel tensor isnt set, cant predict")
        if not self.pixels.dtype == torch.float32:
            raise TypeError("Image data is not in float32!")

    def _check_counts_pixels(self):
        if self.counts_pixels is None:
            raise ValueError("counts pixel tensor isnt set, cant predict")
        if not self.counts_pixels.dtype == torch.float32:
            raise TypeError("Image data is not in float32!")

    def _check_geom(self):
        if self.geom is None:
            raise ValueError("geom tensor isnt set, cant predict")

