
import numpy as np
from scipy.ndimage import binary_erosion
from dials.algorithms.integration import filtering
from resonet.utils import qmags


class IceMasker:
    def __init__(self, shot_det, shot_beam):
        self.wavelen = shot_beam.get_wavelength()
        self.dist = shot_det[0].get_distance()
        self.pixsize = shot_det[0].get_pixel_size()[0]
        #self.cent = shot_det[0].get_beam_centre_pix(shot_beam.get_unit_s0())
        self.Q = qmags.qmags(shot_det, shot_beam)
        # this filter lists the ice ring bounds in units of 1/dstar_squared
        self.ice_filt = filtering.IceRingFilter()
        # so we convert them to units of Q (1/d)
        # we flatten (ravel) the Nx2 array of Q bins, so we can quickly determine
        # from a list of Qs (the detector pixels) which ones are within a bound
        self.ice_qbins = np.sqrt(self.ice_filt.ice_rings).ravel()

        # set the ice ring mask
        self._set_is_ice_pixel()

    def _set_is_ice_pixel(self):
        inds = np.searchsorted(self.ice_qbins, self.Q.ravel())
        self.is_ice_pixel = (inds % 2 == 1).reshape(self.Q.shape)
        self.is_ice_pixel = binary_erosion(self.is_ice_pixel[0], iterations=1)[None]

    def mask(self, shot_det, shot_beam):  # dist, pixsize, wavelen):
        # TODO only do this when shot_det and shot_beam are changing!
        self.Q = qmags.qmags(shot_det, shot_beam)
        self._set_is_ice_pixel()
        #if dist != self.dist or pixsize != self.pixsize or wavelen != self.wavelen:
        #    # TODO reset self.Q here and recompute self.is_ice_pixel
        #    pass
        return self.is_ice_pixel
