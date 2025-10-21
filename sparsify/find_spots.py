
import numpy as np
from scipy import ndimage as nd

from dials.algorithms.spot_finding.factory import SpotFinderFactory
from dials.array_family import flex
from dials.command_line.find_spots import phil_scope


def dials_find_spots(data_img, params=None, trusted_flags=None, global_threshold=0,
                    sigma_background=6, sigma_strong=3, algorithm="dispersion_extended"):
    """

    :param data_img: 2-D numpy array image, or sub-image (region of interest, e.g. a shoebox)
    :param params: instance of stills_process params.spotfinder, see method stills_process_params_from_file
    :param trusted_flags: boolean array, same shape as data_img, True=good pixel, False is bad pixels
    :return:
    """
    if params is None:
        params = phil_scope.extract()
        params.spotfinder.threshold.algorithm=algorithm
        params.spotfinder.threshold.dispersion.global_threshold = global_threshold
        params.spotfinder.threshold.dispersion.sigma_background = sigma_background
        params.spotfinder.threshold.dispersion.sigma_strong = sigma_strong
    if trusted_flags is None:
        trusted_flags = np.ones(data_img.shape, bool)
    thresh = SpotFinderFactory.configure_threshold(params)
    flex_data = flex.double(np.ascontiguousarray(data_img))
    flex_trusted_flags = flex.bool(np.ascontiguousarray(trusted_flags))
    spotmask = thresh.compute_threshold(flex_data, flex_trusted_flags)
    spotmask = spotmask.as_numpy_array()
    lab, nlab = nd.label(spotmask)
    npix_per_ref = nd.sum(spotmask, lab, index=list(range(1, nlab+1)))
    minpix = 1
    if isinstance(params.spotfinder.filter.min_spot_size, int):
        minpix = params.spotfinder.filter.min_spot_size
    maxpix = np.inf
    if isinstance(params.spotfinder.filter.max_spot_size, int):
        maxpix = params.spotfinder.filter.max_spot_size
    bad_ref_labels = np.where( np.logical_or(npix_per_ref < minpix, npix_per_ref > maxpix))[0]
    for i_lab in bad_ref_labels:
        spotmask[lab==i_lab+1] = False

    return spotmask
