
import torch
from scipy.ndimage import binary_dilation
import numpy as np

from resonet.utils.eval_model import load_model, to_tens

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
        dnew = np.sqrt((B - 4) / 18)
        return dnew
    else:
        dnew = .5 * (-b + np.sqrt(sqrt_arg)) / a  # positive root
        return dnew


class ImagePredict:

    def __init__(self, reso_model=None, multi_model=None, ice_model=None, reso_arch=None,
                 multi_arch=None, ice_arch=None, dev="cpu"):
        """

        Parameters
        ----------
        reso_model
        multi_model
        ice_model
        reso_arch
        multi_arch
        ice_arch
        dev
        """
        self.pixels = None  # this is the image tensor, a (512x512) representation of the diffraction shot
        self.geom = None  # the geometry tensor, (1x5) tensor with elements (detdist, wavelen, pixsize, xdim, ydim)
        self._dev = dev
        self._try_load_model("reso", reso_model, reso_arch)
        self._try_load_model("multi", multi_model, multi_arch)
        self._try_load_model("ice", ice_model, ice_arch)
        self._geom_props = ["detdist_mm", "pixsize_mm", "wavelen_Angstrom", "xdim", "ydim"]
        self.mask = None
        self.maxpool_2x2 = torch.nn.MaxPool2d(2, 2)
        self.maxpool_4x4 = torch.nn.MaxPool2d(4, 4)
        self.allowed_quads = {0: "A", 1: "B", 2: "C", 3: "D"}
        self.quads = [1]

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
            model = model.to(self._dev)
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

    @property
    def quads(self):
        return self._quads

    @quads.setter
    def quads(self, val):
        # TODO assert val is iterable
        for v in val:
            if v not in self.allowed_quads:
                raise ValueError("Only values 0,1,2,3 can be set as quads")
        self._quads = [self.allowed_quads[v] for v in val]

    def _set_geom_tensor(self):
        for prop in self._geom_props:
            if getattr(self, prop) is None:
                raise ValueError("Must set %s before initializing geom tensor" % prop)

        self.geom = torch.tensor([[self.detdist_mm, self.pixsize_mm, self.wavelen_Angstrom, self.xdim, self.ydim]])
        self.geom = self.geom.to(self._dev)

    def _set_pixel_tensor(self, raw_img):
        """pass in a raw image (2D array) and convert it to an torch tensor for prediction"""
        # check for mask and set a default if none found
        self._set_default_mask(raw_img)

        is_pil = self.xdim == 2463
        if is_pil:
            dwnsamp = 2
            maxpool = self.maxpool_2x2
        else:  # eiger or some other large format
            maxpool = self.maxpool_4x4
            dwnsamp = 4
        tensors = []
        for quad in self.quads:
            tens = to_tens(raw_img, self.mask, maxpool=maxpool,
                           ds_fact=dwnsamp, quad=quad, dev=self._dev)
            tensors.append(tens)
        self.pixels = torch.concatenate(tensors)

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
        one_over_reso = self.reso_model(self.pixels, self.geom)
        reso = torch.min(1/one_over_reso).item()
        reso = d_to_dnew(reso)
        return reso

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

    def _check_geom(self):
        if self.geom is None:
            raise ValueError("geom tensor isnt set, cant predict")

