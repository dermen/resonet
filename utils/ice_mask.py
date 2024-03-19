
import numpy as np
from scipy.ndimage import binary_erosion
from dials.algorithms.integration import filtering
from dxtbx.model import DetectorFactory, BeamFactory, Detector, Panel, Beam

from resonet.utils import qmags


class IceMasker:
    def __init__(self, dxtbx_geom_dict=None, simple_geom_dict =None):
        """

        :param dxtbx_geom_dict:  a dictionary of DXTBX detector object and DXTBX beam object.
            The dictionary keys should be 'detector' and 'beam'
        :param simple_geom_dict: dictionary where the keys are
            'wavelength_Ang': X-ray wavelength in Angstrom
            'distance_mm': sample to camera length in mm
            'pixsize_mm': size of pixel in mm (assumes square pixels)
            'beam_x': center coordinate of forward beam on image in pixel units (fast-scan direction)
            'beam_y': center coordinate of foreward beam on image in pixel units (slow-scan direction)
            'fast_dim': the fast-scan dimension of the 2D image (integer)
            'slow_dim': the slow-scan dimension of the 2D image (integer)
        """
        if dxtbx_geom_dict is None:
            assert simple_geom_dict is not None, "Need one of dxtbx_geom_dict or simple_geom_dict to be not None"
            self.wavelen = simple_geom_dict["wavelength_Ang"]
            self.dist = simple_geom_dict["distance_mm"]
            self.pixsize = simple_geom_dict["pixsize_mm"]
            self.beam_x = simple_geom_dict["beam_x"]
            self.beam_y = simple_geom_dict["beam_y"]
            self.fast_dim = int(simple_geom_dict["fast_dim"])
            self.slow_dim = int(simple_geom_dict["slow_dim"])
            center = self.beam_x*self.pixsize, self.beam_y*self.pixsize
            shot_det = DetectorFactory.simple("PAD", self.dist, center, "+x", "+y",
                                              (self.pixsize, self.pixsize), (self.fast_dim, self.slow_dim))
            shot_beam = BeamFactory.simple(self.wavelen)
        else:
            assert simple_geom_dict is None, "Need one of dxtbx_geom_dict or simple_geom_dict to be not None"
            shot_det = dxtbx_geom_dict["detector"]
            shot_beam = dxtbx_geom_dict["beam"]
            self.wavelen = shot_beam.get_wavelength()
            self.dist = shot_det[0].get_distance()
            self.pixsize = shot_det[0].get_pixel_size()[0]
            self.beam_x, self.beam_y = shot_det[0].get_beam_centre_px(shot_beam.get_unit_s0())
            self.beam_dict = shot_beam.to_dict()

        # cache the panel and beam dictionaries
        self.panel_dict = shot_det[0].to_dict()
        self.beam_dict = shot_beam.to_dict()

        # define the Q of each pixel
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
        # TODO: change pad width to be a function of pixel solid angle ?
        self.is_ice_pixel = binary_erosion(self.is_ice_pixel[0], iterations=1)[None]

    def mask(self, distance, wavelength, beam_x, beam_y):
        """
        :param distance:  detector distance in mm
        :param wavelength: X-ray wavelength in Angstrom
        :param beam_x: coordinate of forward beam in pixel units (fast-scan)
        :param beam_y: coordinate of foward beam in pixel units (slow-scan)
        :return:
        """
        # TODO track if pixel size and xdim/ydim change because these change depending on the binning mode of the detector
        # check whether the Q of each pixel has changed. Assume detector model doesnt change between runs, so image dimensions and pixel size dont need to be checked
        if not np.allclose([distance, wavelength, beam_x, beam_y], [self.dist, self.wavelen, self.beam_x, self.beam_y]):
            print("Recalculating mask because geom has changed!")
            self.panel_dict["distance"] = distance

            fast_axis = np.array(self.panel_dict["fast_axis"])
            slow_axis = np.array(self.panel_dict["slow_axis"])
            pixsize = self.panel_dict["pixel_size"][0]
            origin = - fast_axis*beam_x*pixsize - slow_axis*beam_y*pixsize - np.array([0,0,-distance])
            self.panel_dict["origin"] = tuple(origin)
            # update the wavelength
            self.beam_dict["wavelength"] = wavelength

            shot_panel = Panel.from_dict(self.panel_dict)
            shot_det = Detector()
            shot_det.add_pannel(shot_panel)
            shot_beam = Beam.from_dict(self.beam_dict)
            self.Q = qmags.qmags(shot_det, shot_beam)
            self._set_is_ice_pixel()
            # update the internal
            self.beam_x = beam_x
            self.beam_y = beam_y
            self.dist = distance
            self.wavelen = wavelength
        else:
            print("Using existing mask!")
        return self.is_ice_pixel
