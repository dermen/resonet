
import sys
from collections import OrderedDict
import numpy as np

from resonet.utils import maxbin

try:
    import torch
    from resonet.params import ARCHES
except ImportError:
    pass


HAS_TORCH = "torch" in sys.modules


def strip_names_in_state(orig_state):
    new_state = OrderedDict()
    for k, v in orig_state.items():
        if k.startswith("module."):
            k = k[7:]
        new_state[k] = v
    return new_state


def load_model(state_name, arch="res50", ori_mode=False):
    assert HAS_TORCH
    assert arch in ARCHES
    kwargs = {"dev": "cpu"}
    if ori_mode:
        kwargs["nout"] = 6
    model = ARCHES[arch](**kwargs)
    model.ori_mode = ori_mode
    temp = torch.load(state_name, map_location=torch.device('cpu'))
    state = OrderedDict()
    for k,v in temp.items():
        if k.startswith("module."):
            k = k[7:]
        state[k] = v
        
    model.load_state_dict(state, strict=False)
    model = model.to("cpu")
    model = model.eval()
    return model
    

# TODO: remove un-used to_tens methods

def raw_img_to_tens(raw_img, MASK, howbin='max', quad="A", cent=None, numpy_only=False,
        adu_per_photon=1, ds_fact=4, IMAX=None):


    img = maxbin.maximg_downsample((raw_img/adu_per_photon)*MASK, factor=ds_fact)
    img[img < 0] = 0
    if IMAX is None:
        IMAX = 255**2
    img[img >= IMAX] = IMAX
    img = np.sqrt(img)
    img = img.astype(np.uint8).astype(np.float32)

    if cent is None:
        y_cent, x_cent = [ int(x/2.) for x in raw_img.shape]
        cent = x_cent, y_cent
    cent_ds = int(round(cent[0]/ds_fact)), int(round(cent[1]/ds_fact))
    x,y = cent_ds
    if quad=="A":
        subimg=img[y-512:y, x-512:x]
        quad = np.rot90(subimg, k=2)
        # optionally pad quad image to be 512 x 512
    elif quad=="B":
        subimg = img[y-512:y, x:x+512]
        quad = np.rot90(subimg, k=3)
    elif quad=="C":
        subimg = img[y:512+y, x-512:x]
        quad = np.rot90(subimg, k=1)
    else: # quad=="D":
        subimg = img[y:512+y, x:512+x]
        quad = subimg

    if HAS_TORCH and not numpy_only:
        quad = torch.tensor(quad.copy()).view((1,1,512,512)).to("cpu")
    return quad


def raw_img_to_tens_pil(raw_img, MASK, xy=None, numpy_only=False, howbin='max'):
    ysl, xsl = maxbin.get_slice_pil(xy)
    # or else pad img if shape is not 1024x1024
    img = maxbin.img2int_pil(raw_img[ysl, xsl]*MASK[ysl, xsl], howbin=howbin)
    img = maxbin.get_quadA_pil(img).astype(np.float32)
    if HAS_TORCH and not numpy_only:
        img = torch.tensor(img.copy()).view((1,1,512,512)).to("cpu")
    return img


def raw_img_to_tens_mar(raw_img, MASK, numpy_only=False, howbin='max'):
    img = maxbin.img2int_mar(raw_img*MASK, howbin=howbin)
    img = maxbin.get_quadA_mar(img).astype(np.float32)
    if HAS_TORCH and not numpy_only:
        img = torch.tensor(img).view((1,1,512,512)).to("cpu")
    return img


def raw_img_to_tens_jung(raw_img, mask, numpy_only=False, cent=None, IMAX=None, adu_per_photon=9.481, quad="B", dev="cpu"):
    assert quad in ["A","B","C","D"]

    if cent is None:
        cent = 2106, 2224  # from swissFEL geom
    ds_fact = 4  # downsample factor
    maxpool = torch.nn.MaxPool2d(ds_fact,ds_fact)
    cent_ds = int(round(cent[0]/ds_fact)), int(round(cent[1]/ds_fact))
    #img = maxbin.maximg_downsample((raw_img/adu_per_photon)*mask, factor=4)
    img = maxbin.maximg_downsample((raw_img/adu_per_photon)*mask, factor=ds_fact, maxpool=maxpool, dev=dev, leave_on_gpu=False, convert_to_f32=True)
    img[img < 0] = 0
    SQRT = np.sqrt
    ROT90 = np.rot90
    FLOOR = np.floor
    if not isinstance(img, np.ndarray):
        SQRT = torch.sqrt
        ROT90 = torch.rot90
        FLOOR = torch.floor
    if IMAX is None:
        IMAX = 2**14
    img[img > IMAX] = IMAX
    img = SQRT(img)
    img = FLOOR(img)
    #img = img.astype(np.uint8).astype(np.float32)

    x,y = cent_ds
    if quad=="A":
        subimg=img[y-512:y, x-512:x]
        quad = ROT90(subimg, k=2)
        # optionally pad quad image to be 512 x 512
    elif quad=="B":
        subimg = img[y-512:y, x:x+512]
        quad = ROT90(subimg, k=3)
    elif quad=="C":
        subimg = img[y:512+y, x-512:x]
        quad = ROT90(subimg, k=1)
    else: # quad=="D":
        subimg = img[y:512+y, x:512+x]
        quad = subimg

    if not numpy_only and isinstance(quad, np.ndarray):
        quad = torch.tensor(quad.copy()).view((1,1,512,512)).to("cpu")

    return quad


def raw_img_to_tens_pil2(raw_img, mask, numpy_only=False, cent=None, IMAX=None, adu_per_photon=1, quad="B", ds_fact=2, sqrt=True, maxpool=None, dev="cpu", leave_on_gpu=False, convert_to_f32 =False):
    assert quad in ["A","B","C","D"]

    if cent is None:
        cent = 1231.5, 1263.5
    cent_ds = int(round(cent[0]/ds_fact)), int(round(cent[1]/ds_fact))
    #if lcn:
    #   from skimage import exposure
    #    new_img = exposure.equalize_adapthist(img*mask, kernel_size=16)

    img = maxbin.maximg_downsample((raw_img/adu_per_photon)*mask, factor=ds_fact, maxpool=maxpool, dev=dev, leave_on_gpu=leave_on_gpu, convert_to_f32=convert_to_f32)
    img[img < 0] = 0
    if IMAX is None:
        IMAX = 255**2
    img[img >= IMAX] = IMAX
    if sqrt:
        if leave_on_gpu:
            img = torch.sqrt(img)
        else:
            img = np.sqrt(img)
            img = img.astype(np.uint8)
    else:
        if leave_on_gpu:
            img = torch.floor(img)
        else:
            img = img.astype(np.int32)
    if not leave_on_gpu:
        img = img.astype(np.float32)

    x,y = cent_ds
    n = 512 #1024//ds_fact
    rot90 = np.rot90
    if leave_on_gpu:
        rot90 = torch.rot90
    if quad=="A":
        subimg=img[y-n:y, x-n:x]
        quad = rot90(subimg, k=2)
        # optionally pad quad image to be 512 x 512
    elif quad=="B":
        subimg = img[y-n:y, x:x+n]
        quad = rot90(subimg, k=3)
    elif quad=="C":
        subimg = img[y:n+y, x-n:x]
        quad = rot90(subimg, k=1)
    else: # quad=="D":
        subimg = img[y:n+y, x:n+x]
        quad = subimg

    if HAS_TORCH and not numpy_only:
        if leave_on_gpu:
            quad= quad[None,None]
        else:
            quad = torch.tensor(quad.copy()).view((1,1,n,n)).to("cpu")

    return quad


def to_tens(raw_img, mask, maxpool, cent=None, maxval=65025, adu_per_photon=1,
            quad="A", ds_fact=2, sqrt=True, dev="cpu", convert_to_f32 =False):
    """

    Parameters
    ----------
    raw_img: np.ndarray, image array (numpy), e.g. 2527 x 2463 for Pilatius 6M
    mask: np.ndarray, image mask, same shape as raw_img, but boolean with 1,0 representing unmasked,masked
    maxpool: instance of torch.MaxPool2d
    cent: 2-tuple of float, optional center of camera, (fast-scan coordinate, slow-scan coordinate)
    maxval: float, saturation value (zero-out pixels above this value
    adu_per_photon: float, factor to convert raw_img to photon units
    quad: str, which of the 4 quads to extract ('A','B','C', or 'D')
    ds_fact: int, what factor to downsample the image (final result should be close to 512 x 512 per quad, so set to 2
        for Pilatus 6M and 4 for Eiger 16M
    sqrt: bool, whether to apply sqrt to data as a normalization procedure
    dev: pytorch device for resulting tensor to be allocated on
    convert_to_f32: convert to float32 precision

    Returns
    -------
    downsampled image stored in a 4-dimensional torch.Tensor
    """
    assert quad in ["A", "B", "C", "D"]
    assert len(raw_img.shape) == 2

    if cent is None:  # if center is not provided, default to
        y_cent, x_cent = [x/2. for x in raw_img.shape]
        cent = x_cent, y_cent  # e.g., (1231.5, 1263.5) for Pilatus 6M

    n = 512 * ds_fact  # final quad dimension is always 512
    x, y = int(round(cent[0])), int(round(cent[1]))
    if quad == "A":
        subimg = raw_img[y-n:y, x-n:x]
        submask = mask[y-n:y, x-n:x]
        k = 2
    elif quad=="B":
        subimg = raw_img[y-n:y, x:x+n]
        submask = mask[y-n:y, x:x+n]
        k = 3
    elif quad == "C":
        subimg = raw_img[y:n+y, x-n:x]
        submask = mask[y:n+y, x-n:x]
        k = 1
    else:
        subimg = raw_img[y:n+y, x:n+x]
        submask = mask[y:n+y, x:n+x]
        k = 0

    masked_subimg = (subimg/adu_per_photon)*submask

    if maxpool is None:
        if convert_to_f32 and not quad.dtype == np.float32:
            masked_subimg = masked_subimg.astype(np.float32)
        quad = maxbin.maximg_downsample(masked_subimg, factor=ds_fact)
        quad = torch.tensor(quad).to(dev)

    else:
        quad = torch.tensor(masked_subimg).to(dev)
        if convert_to_f32 and not quad.dtype == torch.float32:
            quad = quad.astype(torch.float32)
        quad = maxbin.downsample_tensor(quad, ds_fact, maxpool)

    if k > 0:
        quad = torch.rot90(quad, k=k)

    quad[quad < 0] = 0
    quad[quad >= maxval] = maxval

    if sqrt:
        quad = torch.sqrt(quad)
    quad = torch.floor(quad)

    quad = quad[None, None]

    return quad

