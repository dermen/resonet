import numpy as np
from simtbx.diffBragg.utils import image_data_from_expt


def project_jungfrau(expt, normalize=True, mask=None, return_center=False, img=None):
    """
    Note, this works for any multi-panel detector, historically its called project_jungfrau
    :param expt: dxtbx experiment object
    :param normalize: whether to use mean (versus sum) for binning
    :param mask: optional mask
    :return: returns 2D projection of image data from expt
    """
    P, F, S = make_psf(expt.detector)
    all_coords = []
    if img is None:
        img = image_data_from_expt(expt)
    _, Ydim, Xdim = img.shape
    Jcoord, Icoord = np.indices((Ydim, Xdim))

    for orig, fast, slow in zip(P, F, S):
        coords = Icoord[:, :, None] * fast + Jcoord[:, :, None] * slow + orig
        all_coords.append(coords)
    all_coords = np.array(all_coords)

    all_X = np.round(all_coords[:, :, :, 0]).astype(int)
    min_X = all_X.min()
    all_X -= min_X

    all_Y = np.round(all_coords[:, :, :, 1]).astype(int)
    min_Y = all_Y.min()
    all_Y -= min_Y
    max_Y = all_Y.max()
    max_X = all_X.max()

    proj = np.zeros((int(max_Y) + 1, int(max_X) + 1))
    proj_sh = proj.shape
    inds = (proj_sh[1] * all_Y.ravel() + all_X.ravel())
    if mask is not None:
        np.add.at(proj.ravel(), inds, mask.astype(img.dtype).ravel())
    else:
        np.add.at(proj.ravel(), inds, img.ravel())
    if normalize:
        norm = np.zeros_like(proj)
        np.add.at(norm.ravel(), inds, np.ones_like(inds))
        with np.errstate(invalid='ignore'):
            proj = np.nan_to_num(proj/norm)

    cent = abs(min_X), abs(min_Y)
    if return_center:
        return proj, cent
    else:
        return proj


def make_psf(DET):
    """

    :param DET:  dxtbx detector model
    :return: 3 arrays, one is the panel origin, then panel slow vectors, then panel fast vectors
    """
    P, S, F = [], [], []
    for i in range(len(DET)):
        panel = DET[i]
        origin = np.array(panel.get_origin())
        fdet = np.array(panel.get_fast_axis())
        # fdet = np.array([fdet[0], fdet[1], 0])
        sdet = np.array(panel.get_slow_axis())
        # sdet = np.array([sdet[0], sdet[1], 0])
        # fdet /= np.linalg.norm(fdet)
        # sdet /= np.linalg.norm(sdet)
        pixsize = panel.get_pixel_size()[0]
        P.append(origin / pixsize)
        S.append(sdet)
        F.append(fdet)

    return np.array(P),np.array(F),np.array(S)

