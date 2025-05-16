
import numpy as np
import h5py
import ast

from dxtbx.format.FormatHDF5 import FormatHDF5
from dials.array_family import flex


class FormatDiffComp(FormatHDF5):
    """
    Class for reading HDF5 files for arbitrary geometries
    focused on performance
    """
    @staticmethod
    def understand(image_file):
        try:
            img_handle = h5py.File(image_file, "r")
        except (IOError, OSError):
            return False
        if "format" not in img_handle.attrs:
            return False
        if img_handle.attrs["format"] != "DiffComp":
            return False
        return True

    def _start(self):
        self._handle = h5py.File(self._image_file, "r")
        self.images = list(self._handle.keys())
        self._geometry_define()

    def _geometry_define(self):
        det_str = self._handle.attrs["dxtbx_detector_string"]
        beam_str = self._handle.attrs["dxtbx_beam_string"]
        is_rot = "dxtbx_gonio_string" in self._handle.attrs
        if is_rot:
            gonio_str = self._handle.attrs["dxtbx_gonio_string"]
            scan_str = self._handle.attrs["dxtbx_scan_string"]
        try:
            det_str = det_str.decode()
            beam_str = beam_str.decode()
            if is_rot:
                gonio_str = gonio_str.decode()
                scan_str = scan_str.decode()
        except AttributeError:
            pass
        det_dict = ast.literal_eval(det_str)
        beam_dict = ast.literal_eval(beam_str)
        self._cctbx_detector = self._detector_factory.from_dict(det_dict)
        self._cctbx_beam = self._beam_factory.from_dict(beam_dict)

        npanel = len(self._cctbx_detector)
        fdim, sdim = self._cctbx_detector[0].get_image_size()
        self.img_shape = npanel, sdim, fdim
        #dtype = self.images[0]["vals"].dtype
        self.panels = np.zeros(self.img_shape, dtype=np.float64)#.astype(dtype)

        if is_rot:
            gonio_dict = ast.literal_eval(gonio_str)
            scan_dict = ast.literal_eval(scan_str)
            self._cctbx_gonio = self._goniometer_factory.from_dict(gonio_dict)
            self._cctbx_scan = self._scan_factory.from_dict(scan_dict)

    def get_num_images(self):
        return len(self.images)

    def get_raw_data(self, index=0):
        img_key = self.images[index]
        img_group = self._handle[img_key]
        fast = img_group["fast"][()]
        slow = img_group["slow"][()]
        pid = img_group["panel"][()]
        vals = img_group["vals"][()]
        self.panels *= 0
        self.panels[pid, slow, fast] = vals

        if self.panels.dtype == np.float64:
            flex_data = [flex.double(p) for p in self.panels]
        else:
            flex_data = [flex.double(p.astype(np.float64)) for p in self.panels]
        return tuple(flex_data)

    def get_detectorbase(self, index=None):
        raise NotImplementedError

    def get_detector(self, index=None):
        return self._cctbx_detector

    def get_goniometer(self):
        return self._cctbx_gonio

    def get_scan(self):
        return self._cctbx_scan

    def get_beam(self, index=0):
        return self._cctbx_beam


if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        print(FormatDiffComp.understand(arg))
