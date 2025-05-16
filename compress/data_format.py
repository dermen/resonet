
import h5py
import numpy as np
import os
import json


class DiffCompWriter:

  def __init__(self, filename, detector, beam, dtype=None, compression_args=None,
               goniometer=None, scan=None, file_ops=None):
    """
    Simple class for writing dxtbx compatible HDF5 files

    :param filename:  input file path
    :param detector: dxtbx detector model
    :param beam: dxtbx beam model
    :param dtype: datatype for storage
    :param compression_args: compression arguments for h5py, lzf is performant and simple
        if you only plan to read file in python
        Examples:
          compression_args={"compression": "lzf"}  # Python only
          comression_args = {"compression": "gzip", "compression_opts":9}
    :param goniometer: dxtbx goniometer obj
    :param scan: dxtbx scan obj
    """
    if file_ops is None:
        file_ops = {}
    self.compresion_args = {} if compression_args is None else compression_args
    self.file_handle = h5py.File(filename, 'w', **file_ops)
    self.beam = beam
    self.detector = detector
    self.goniometer = goniometer
    self.scan = scan
    if dtype is None:
      dtype = np.float32
    self.dtype = dtype
    self._write_geom()
    self.dtype = dtype
    self.file_handle.attrs["format"] = "DiffComp"

  def add_image(self, pid, fast, slow, val, key):
    """
    :param image: a single image as numpy image, same shape as used to instantiate the class
    """
    new_keys = [os.path.join(key, name) for name in ["panel", "fast", "slow", "vals"]]
    self.file_handle.create_dataset(new_keys[0], data=pid, dtype=np.uint16, **self.compresion_args)
    self.file_handle.create_dataset(new_keys[1], data=fast, dtype=np.uint16, **self.compresion_args)
    self.file_handle.create_dataset(new_keys[2], data=slow, dtype=np.uint16, **self.compresion_args)
    self.file_handle.create_dataset(new_keys[3], data=val, dtype=self.dtype, **self.compresion_args)

  def _write_geom(self):
    beam = self.beam
    det = self.detector
    if not isinstance(beam, dict):
      beam = beam.to_dict()
    if not isinstance(det, dict):
      det = det.to_dict()
    self.file_handle.attrs['dxtbx_beam_string'] = json.dumps(beam)
    self.file_handle.attrs['dxtbx_detector_string'] = json.dumps(det)

    if self.goniometer is not None:
      gonio = self.goniometer
      if not isinstance(gonio, dict):
        gonio = gonio.to_dict()
      self.file_handle.attrs['dxtbx_gonio_string'] = json.dumps(gonio)
      scan = self.scan
      if not isinstance(scan, dict):
        scan = scan.to_dict()
        self.file_handle.attrs['dxtbx_scan_string'] = json.dumps(scan)

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.file_handle.close()

  def __enter__(self):
    return self

  def close_file(self):
    """
    close the file handle (if instantiated using `with`, then this is done automatically)
    """
    self.file_handle.close()

