# dxtbx beam model description
beam_descr = {'direction': (-1,0,0),
             'divergence': 0.0,
             'flux': 4e11,
             'polarization_fraction': 1.,
             'polarization_normal': (0.0, 1.0, 0.0),
             'sigma_divergence': 0.0,
             'transmission': 1.0,
             'wavelength': 0.977794 }

# monolithic camera description
det_descr = {'panels':
               [{'fast_axis': (0.0, 0.0, 1.0),
                 'gain': 1.0,
                 'identifier': '',
                 'image_size': (4150, 4371),
                 'mask': [],
                 'material': '',
                 'mu': 0.0,
                 'name': 'Panel',
                 'origin': (200, 163.912, -155.625),
                 'pedestal': 0.0,
                 'pixel_size': (0.075, 0.075),
                 'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
                 'raw_image_offset': (0, 0),
                 'slow_axis': (0.0, -1.0, 0.0),
                 'thickness': 0.0,
                 'trusted_range': (0.0, 65536.0),
                 'type': ''}]}


from dxtbx.model.detector import DetectorFactory
from dxtbx.model.beam import BeamFactory

DET = DetectorFactory.from_dict(det_descr)
BEAM = BeamFactory.from_dict(beam_descr)
