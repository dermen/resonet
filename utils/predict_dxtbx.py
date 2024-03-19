
import numpy as np
import dxtbx
from resonet.utils.predict import ImagePredict


class ImagePredictDxtbx(ImagePredict):
    """

    Example usage:
    >> impred = ImagePredictDxtbx(reso_model="reso_ep102.nn", reso_arch="res50",
        multi_model="multi_ep77.nn", multi_arch="res34")
    >> impred.load_image_from_file("something.cbf")
    >> reso = impred.detect_resolution()
    >> is_multi = impred.detect_multilattice()
    >> impred.load_image_from_file("something_else.mccd")
    >> another_reso = impred.detect_resolution()
    """

    def __init__(self, *args, **kwargs):
        """
        performant image reader, specializing in square monolithic cameras
        Use this for most synchrotron data

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    def load_image_from_file(self, image_file, filenum=0, use_ice_mask=False):
        """
        :param image_file:  path to an image file readable by DXTBX
        :param use_ice_mask: bool, whether or not to add ice rings to the loaded image
        """
        loader = dxtbx.load(image_file)
        try:
            raw_image = loader.get_raw_data()
            file_num_req = False
        except:  # TODO put proper exception here
            raw_image = loader.get_raw_data(filenum)
            file_num_req = True

        if file_num_req:
            det = loader.get_detector(filenum)
            beam = loader.get_beam(filenum)
        else:
            det = loader.get_detector()
            beam = loader.get_beam()

        if isinstance(raw_image, tuple):
            raw_image = np.array([panel.as_numpy_array() for panel in raw_image])
        else:
            raw_image = raw_image.as_numpy_array()

        if len(raw_image.shape) == 3:
            if len(det) > 1:
                raise NotImplementedError("Not currently supporting multi panel formats")
            raw_image = raw_image[0]
        if not raw_image.dtype == np.float32:
            raw_image = raw_image.astype(np.float32)

        self.xdim, self.ydim = det[0].get_image_size()
        self.pixsize_mm = det[0].get_pixel_size()[0]
        self.detdist_mm = abs(det[0].get_distance())
        self.wavelen_Angstrom = beam.get_wavelength()
        self._set_geom_tensor()
        if use_ice_mask:
            dxtbx_geom = {"detector":det, "beam": beam}
            self.set_ice_mask(dxtbx_geom=dxtbx_geom)
        self._set_pixel_tensor(raw_image)
