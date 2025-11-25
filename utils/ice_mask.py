
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
try:
    from dxtbx.model import DetectorFactory, BeamFactory, Detector, Panel, Beam
    has_dxtbx=True
except ImportError:
    has_dxtbx = False

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
        assert has_dxtbx
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
        self.ice_filt_rings = [(0.06293371930015712, 0.06940133881989617),
            (0.07151283840467029, 0.07801859052119219),
            (0.08153575314298465, 0.08808140832075168),
            (0.13741309889148712, 0.1441238503192669),
            (0.19477808388384213, 0.20160375689547588),
            (0.2306296306687546, 0.2375120718109308),
            (0.2607677453000252, 0.26769187663751054),
            (0.26937182872252763, 0.27630703725783184),
            (0.27941880139664704, 0.2863665340047717),
            (0.2951853373124733, 0.3021518995695604),
            (0.3353772161087531, 0.34238782388447514),
            (0.3611943712806426, 0.36823072959700354),
            (0.4286534637006733, 0.435749724620467),
            (0.45880308470771586, 0.46592329451887254),
            (0.4774602054318337, 0.48459450357604406),
            (0.4932313585342249, 0.5003771670543299),
            (0.5291037804383912, 0.536274529293368),
            (0.533433310255756, 0.5406069606730568),
            (0.5592559078474261, 0.5664464161760849),
            (0.5908518402068054, 0.5980620039324418),
            (0.6095107012760419, 0.616732011590955),
            (0.6267264876956948, 0.6339578004607493),
            (0.6654883450715705, 0.6727412611411309),
            (0.6683232966714301, 0.6755777457649381),
            (0.7271890295898772, 0.7344739963176857),
            (0.7343540197431445, 0.7416425403352668),
            (0.7573440555217097, 0.7646437620305668),
            (0.758787391400734, 0.7660877894209114),
            (0.7889427389924608, 0.7962573073090558),
            (0.8549761456946199, 0.8623200355309848),
            (0.8635847987573375, 0.8709323522106965),
            (0.8664199391284152, 0.8737686915850057),
            (0.8736368552585424, 0.8809886431406379),
            (0.8894108713456698, 0.8967692120223426),
            (0.9252892878777872, 0.9326621328252107),
            (0.9296194968709873, 0.936994056233382),
            (0.932454681985946, 0.9398303597657177),
            (0.9769423206084237, 0.9843351387968259)]

        # so we convert them to units of Q (1/d)
        # we flatten (ravel) the Nx2 array of Q bins, so we can quickly determine
        # from a list of Qs (the detector pixels) which ones are within a bound
        self.ice_qbins = np.sqrt(self.ice_filt_rings).ravel()

        # set the ice ring mask
        self._set_is_ice_pixel()

    def _set_is_ice_pixel(self):
        inds = np.searchsorted(self.ice_qbins, self.Q.ravel())
        self.is_ice_pixel = (inds % 2 == 1).reshape(self.Q.shape)
        # TODO: change pad width to be a function of pixel solid angle ?
        #self.is_ice_pixel = binary_erosion(self.is_ice_pixel[0], iterations=1)[None]
        self.is_ice_pixel = binary_dilation(self.is_ice_pixel[0], iterations=1)[None]

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
            shot_det.add_panel(shot_panel)
            shot_beam = Beam.from_dict(self.beam_dict)
            self.Q = qmags.qmags(shot_det, shot_beam)
            self._set_is_ice_pixel()
            # update the internal
            self.beam_x = beam_x
            self.beam_y = beam_y
            self.dist = distance
            self.wavelen = wavelength
        #else:
        #    print("Using existing mask!")
        return self.is_ice_pixel
