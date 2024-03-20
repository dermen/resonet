
import fabio
from resonet.utils.predict import ImagePredict


class ImagePredictFabio(ImagePredict):

    def __init__(self, *args, **kwargs):
        """
        performant image reader, specializing in square monolithic cameras
        Use this for most synchrotron data

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    def load_image_from_file_or_array(self, detdist=None, pixsize=None, wavelen=None, image_file=None, raw_image=None,
                                      beam_center=None, use_ice_mask=False):
        """
        :param detdist: sample-to-detector distance in mm
        :param pixsize: pixel size in mm
        :param wavelen: wavelength in Angstrom
        :param image_file:  path to an image file readable by FABIO (most CBF and MCCD should work)
        :param raw_image:  2D numpy array
        :param beam_center: 2-tuple representing direct beam coordinate on the image in pixel units (fast-scan coord, slow-scan coord)
        :param use_ice_mask: boolean, whether to apply the ice mask, requires all the geometry arguments detdist, pixsize, wavelen, and beam_center
        """
        if image_file is None:
            assert raw_image is not None, "Need a raw image or an image file!"
        else:
            raw_image = self.get_image_array(image_file)

        # set the simple geometry
        self.ydim, self.xdim = raw_image.shape
        self.detdist_mm = detdist
        self.pixsize_mm = pixsize
        self.wavelen_Angstrom = wavelen
        if not any([val is None for val in [detdist, pixsize, wavelen]]):
            self._set_geom_tensor()
        if use_ice_mask:
            if any([x is None for x in [beam_center, detdist, wavelen, pixsize]]):
                raise ValueError("geometry variables beam_center, detdist, wavelen, pixsize must be not None for ice masking")
            if not isinstance(beam_center, tuple):
                raise TypeError("beam center needs to be a tuple of len 2")
            beam_x,beam_y = beam_center
            slow_dim, fast_dim = raw_image.shape
            from IPython import embed;embed()
            simple_geom = {"wavelength_Ang": wavelen, "distance_mm": detdist, "pixsize_mm": pixsize,
                           "beam_x": beam_x, "beam_y": beam_y, "fast_dim": fast_dim, "slow_dim": slow_dim}
            self.set_ice_mask(simple_geom=simple_geom)
        self._set_pixel_tensor(raw_image)

    @staticmethod
    def get_image_array(image_file):
        return fabio.open(image_file).data