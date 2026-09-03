import numpy as np
from dxtbx.model import Panel, Detector
import numpy as np


def get_ideal_slices(region_slices, image_shape):
    # 1. Find the consistent Tile H/W (ignoring the beamstop-affected ones)
    dims_y = [s[0].stop - s[0].start for s in region_slices if (s[0].stop - s[0].start) > 0]
    dims_x = [s[1].stop - s[1].start for s in region_slices if (s[1].stop - s[1].start) > 0]
    tile_h = int(np.median(dims_y))
    tile_w = int(np.median(dims_x))

    # 2. Find the stride (the distance from start of one tile to start of next)
    # Using np.diff on unique starts gives us the consistent pitch
    y_starts = sorted(list(set(s[0].start for s in region_slices)))
    x_starts = sorted(list(set(s[1].start for s in region_slices)))

    # Calculate step (pitch)
    step_y = int(np.median(np.diff(y_starts)))
    step_x = int(np.median(np.diff(x_starts)))

    # 3. Generate only what fits
    ideal_slices = []
    max_y, max_x = image_shape

    # We assume a fixed grid layout (e.g., 4 columns)
    num_cols = len(x_starts)

    current_y = 0
    while current_y + tile_h <= max_y:
        for c in range(num_cols):
            x_start = c * step_x
            ideal_slices.append((
                slice(current_y, current_y + tile_h),
                slice(x_start, x_start + tile_w)
            ))
        current_y += step_y

    return ideal_slices


def split_eiger_16M_to_panels(raw, detector=None):
    """

    :param raw: raw 2D eiger 16M image
    :param detector: dxtbx detector model for monolithic eiger - will be converted to a 32 panel detector model if provided
    :return: bunch of stuff
    """
    from scipy.ndimage import label, find_objects
    regions, nregions = label(raw >= 0)
    assert nregions in {32,60}, "nregions=%d" % nregions
    region_slices = find_objects(regions)
    region_slices = get_ideal_slices(region_slices, raw.shape)
    panels = []
    new_detector = Detector()

    for sY, sX in region_slices:
        assert (sY.stop - sY.start) in {512,514, 195}
        assert (sX.stop - sX.start) in {1028,1030, 487}
        raw_panel = raw[sY, sX]
        pad_eiger = False
        if raw_panel.shape==(514,1030):
            pad_eiger=True
            raw_panel = raw_panel[1:-1, 1:-1]
        panels.append(raw_panel)
        if detector is not None:
            pan_dict = detector[0].to_dict()
            orig = np.array(pan_dict["origin"])
            pixsize = pan_dict["pixel_size"][0]
            fast = np.array(pan_dict["fast_axis"])
            slow = np.array(pan_dict["slow_axis"])
            if pad_eiger:
                new_orig = orig + fast * pixsize * np.array([sX.start+1, 0, 0]) + slow * pixsize * np.array(
                    [0, sY.start+1, 0])
            else:
                new_orig = orig + fast*pixsize*np.array([sX.start,0,0]) + slow*pixsize*np.array([0,sY.start,0])
            pan_ydim, pan_xdim = raw_panel.shape
            new_image_size = pan_xdim, pan_ydim
            pan_dict["origin"] = tuple(new_orig)
            pan_dict["image_size"] = new_image_size
            pan_dict["mask"] = []
            new_panel = Panel.from_dict(pan_dict)
            new_detector.add_panel(new_panel)

    ret_val = regions, nregions, region_slices, panels
    if detector is not None:
        ret_val += (new_detector,)
    return ret_val


def project_jungfrau(expt, normalize=True, mask=None, return_center=False, img=None):
    """
    Note, this works for any multi-panel detector, historically its called project_jungfrau
    :param expt: dxtbx experiment object
    :param normalize: whether to use mean (versus sum) for binning
    :param mask: optional mask
    :return: returns 2D projection of image data from expt
    """
    from simtbx.diffBragg.utils import image_data_from_expt
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

