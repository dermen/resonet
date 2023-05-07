
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
        super().__init__(self, *args, **kwargs)

    def load_image_from_file_or_array(self, detdist, pixsize, wavelen, image_file=None, raw_image=None):
        """
        :param detdist: sample-to-detector distance in mm
        :param pixsize: pixel size in mm
        :param wavelen: wavelength in Angstrom
        :param image_file:  path to an image file readable by FABIO (most CBF and MCCD should work)
        :param raw_image:  2D numpy array
        """
        if image_file is None:
            assert raw_image is not None, "Need a raw image or an image file!"
        else:
            raw_image = fabio.open(image_file).data

        # set the simple geometry
        self.ydim, self.xdim = raw_image.shape
        self.detdist_mm = detdist
        self.pixsize_mm = pixsize
        self.wavelen_Angstrom = wavelen
        self._set_geom_tensor()
        self._set_pixel_tensor(raw_image)
