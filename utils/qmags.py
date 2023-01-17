
import numpy as np


def qmags(detector, beam):
    """
    detector: dxtbx detector model
    beam: dxtbx beam model
    """
    fastdim, slowdim = detector[0].get_image_size()
    Qmags = np.zeros((len(detector), slowdim, fastdim))
    for pid in range(len(detector)):
        FAST = np.array(detector[pid].get_fast_axis())
        SLOW = np.array(detector[pid].get_slow_axis())
        ORIG = np.array(detector[pid].get_origin())

        Ypos, Xpos = np.indices((slowdim, fastdim))
        px = detector[pid].get_pixel_size()[0]
        Ypos = Ypos* px
        Xpos = Xpos*px

        SX = ORIG[0] + FAST[0]*Xpos + SLOW[0]*Ypos
        SY = ORIG[1] + FAST[1]*Xpos + SLOW[1]*Ypos
        SZ = ORIG[2] + FAST[2]*Xpos + SLOW[2]*Ypos

        Snorm = np.sqrt(SX**2 + SY**2 + SZ**2)

        SX /= Snorm
        SY /= Snorm
        SZ /= Snorm

        bx,by,bz = beam.get_unit_s0()
        QX = (SX - bx) / beam.get_wavelength()
        QY = (SY - by) / beam.get_wavelength()
        QZ = (SZ - bz) / beam.get_wavelength()
        Qmags[pid] = np.sqrt(QX**2 + QY**2 + QZ**2)

    return Qmags


def convert_res_img(res_img):
    img = maxbin.bin_ndarray(res_img[:4360], maxbin.SHAPE_MAXFILT, 'mean')
    return img[:maxbin.SHAPE_CROP[0], :maxbin.SHAPE_CROP[1]]
