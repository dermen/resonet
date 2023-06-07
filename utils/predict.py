
import torch
from scipy.ndimage import binary_dilation

from resonet.utils.eval_model import load_model, raw_img_to_tens_pil, raw_img_to_tens

"""
"""


class ImagePredict:

    def __init__(self, reso_model=None, multi_model=None, ice_model=None, reso_arch=None,
                 multi_arch=None, ice_arch=None):
        self.pixels = None  # this is the image tensor, a (512x512) representation of the diffraction shot
        self.geom = None  # the geometry tensor, (1x5) tensor with elements (detdist, wavelen, pixsize, xdim, ydim)
        self._try_load_model("reso", reso_model, reso_arch)
        self._try_load_model("multi", multi_model, multi_arch)
        self._try_load_model("ice", ice_model, ice_arch)
        self._geom_props = ["detdist_mm", "pixsize_mm", "wavelen_Angstrom", "xdim", "ydim"]
        self.mask = None

    def _try_load_model(self, model_name, model_path, model_arch):
        """
        If model_path is None, model will have a value of None and prediction will be disabled.
        Otherwise, the model will be set.
        :param model_name: class attribute name of the model
        :param model_path: path to the model state file (.nn)
        :param model_arch: name of the model arch (see resonet.parameters or resonet.arches)
        :return:
        """
        model = None
        if model_path is not None:
            if model_arch is None:
                raise ValueError("Arch string required for loading model %s!" % model_name)
            model = load_model(model_path, model_arch)
        setattr(self, "%s_model" % model_name, model)

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, val):
        self._pixels = val

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

    def _set_geom_tensor(self):
        for prop in self._geom_props:
            if getattr(self, prop) is None:
                raise ValueError("Must set %s before initializing geom tensor" % prop)

        self.geom = torch.tensor([[self.detdist_mm, self.pixsize_mm, self.wavelen_Angstrom, self.xdim, self.ydim]])

    def _set_pixel_tensor(self, raw_img):
        """pass in a raw image (2D array) and convert it to an torch tensor for prediction"""
        # check for mask and set a default if none found
        self._set_default_mask(raw_img)

        is_pil = self.xdim == 2463
        if is_pil:
            tens = raw_img_to_tens_pil(raw_img, self.mask)
        else:
            tens = raw_img_to_tens(raw_img, self.mask)
        self._pixels = tens

    def _set_default_mask(self, raw_img):
        if self.mask is None:
            mask = raw_img >= 0
            mask = ~binary_dilation(~mask, iterations=1)
            self.mask = mask

    def detect_resolution(self):
        """
        :return: an estimate of the crystal resolution in Angstroms
        """
        self._check_pixels()
        self._check_geom()
        self._check_model("reso")
        one_over_reso = self.reso_model(self.pixels, self.geom).item()
        return 1/one_over_reso

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
        if getattr(self, "%s_model" % model_name) is None:
            raise ValueError("{model_name}_model path was not provided when instantiating this class. "
                             "Cant detect `{model_name}` scattering!".format(model_name=model_name))

    def _check_pixels(self):
        if self.pixels is None:
            raise ValueError("pixel tensor isnt set, cant predict")

    def _check_geom(self):
        if self.geom is None:
            raise ValueError("geom tensor isnt set, cant predict")

