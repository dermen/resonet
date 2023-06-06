
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

MODEL_A = "/data/blstaff/xtal/mwilson_data/diff_AI/nety_ep40.nn"
MODEL_B = "/global/cscratch1/sd/dermen/3p15_noMulti/trial.2/nety_ep20.nn"


def strip_names_in_state(orig_state):
    new_state = OrderedDict()
    for k, v in orig_state.items():
        if k.startswith("module."):
            k = k[7:]
        new_state[k] = v
    return new_state


def load_model(state_name, arch="res50"):
    assert HAS_TORCH
    assert arch in ARCHES
    model = ARCHES[arch](dev="cpu")
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
    

#MASK = np.load("/data/blstaff/xtal/mwilson_data/mask_mwils.npy")

def raw_img_to_tens(raw_img, MASK, howbin='max'):
    img = maxbin.get_quadA(maxbin.img2int(raw_img*MASK, howbin=howbin))
    img = img.astype(np.float32)[:512,:512]
    if HAS_TORCH:
        img = torch.tensor(img).view((1,1,512,512)).to("cpu")
    return img


def raw_img_to_tens_pil(raw_img, MASK, xy=None, numpy_only=False, howbin='max'):
    ysl, xsl = maxbin.get_slice_pil(xy)
    # or else pad img if shape is not 1024x1024
    img = maxbin.img2int_pil(raw_img[ysl, xsl]*MASK[ysl, xsl], howbin=howbin)
    img = maxbin.get_quadA_pil(img).astype(np.float32)
    if HAS_TORCH and not numpy_only:
        img = torch.tensor(img).view((1,1,512,512)).to("cpu")
    return img


def raw_img_to_tens_mar(raw_img, MASK, numpy_only=False, howbin='max'):
    img = maxbin.img2int_mar(raw_img*MASK, howbin=howbin)
    img = maxbin.get_quadA_mar(img).astype(np.float32)
    if HAS_TORCH and not numpy_only:
        img = torch.tensor(img).view((1,1,512,512)).to("cpu")
    return img


def raw_img_to_tens_pil2(raw_img, mask, numpy_only=False, cent=None, IMAX=None, adu_per_photon=1, quad="B"):
    assert quad in ["A","B","C","D"]

    if cent is None:
        cent = 1231.5, 1263.5
    ds_fact = 2  # downsample factor
    cent_ds = int(round(cent[0]/ds_fact)), int(round(cent[1]/ds_fact))
    img = maxbin.maximg_downsample((raw_img/adu_per_photon)*mask, factor=ds_fact)
    img[img < 0] = 0
    if IMAX is None:
        IMAX = 255**2
    img[img >= IMAX] = IMAX
    img = np.sqrt(img)
    img = img.astype(np.uint8).astype(np.float32)

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


def raw_img_to_tens_jung(raw_img, mask, numpy_only=False, cent=None, IMAX=None, adu_per_photon=9.481, quad="B"):
    assert quad in ["A","B","C","D"]

    if cent is None:
        cent = 2106, 2224  # from swissFEL geom
    ds_fact = 4  # downsample factor
    cent_ds = int(round(cent[0]/ds_fact)), int(round(cent[1]/ds_fact))
    img = maxbin.maximg_downsample((raw_img/adu_per_photon)*mask, factor=4)
    img[img < 0] = 0
    if IMAX is None:
        IMAX = 2**14  #14**2
    img[img > IMAX] = IMAX
    img = np.sqrt(img)
    img = img.astype(np.uint8).astype(np.float32)

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



def raw_img_to_tens_pil2(raw_img, mask, numpy_only=False, cent=None, IMAX=None, adu_per_photon=1, quad="B", ds_fact=2, sqrt=True):
    assert quad in ["A","B","C","D"]

    if cent is None:
        cent = 1231.5, 1263.5
    cent_ds = int(round(cent[0]/ds_fact)), int(round(cent[1]/ds_fact))
    #if lcn:
    #   from skimage import exposure
    #    new_img = exposure.equalize_adapthist(img*mask, kernel_size=16)

    img = maxbin.maximg_downsample((raw_img/adu_per_photon)*mask, factor=ds_fact)
    img *= mask
    img[img < 0] = 0
    if IMAX is None:
        IMAX = 255**2
    img[img >= IMAX] = IMAX
    if sqrt:
        img = np.sqrt(img)
        img = img.astype(np.uint8)
    else:
        img = img.astype(np.int32)
    img = img.astype(np.float32)

    x,y = cent_ds
    n = 1024//ds_fact
    if quad=="A":
        subimg=img[y-n:y, x-n:x]
        quad = np.rot90(subimg, k=2)
        # optionally pad quad image to be 512 x 512
    elif quad=="B":
        subimg = img[y-n:y, x:x+n]
        quad = np.rot90(subimg, k=3)
    elif quad=="C":
        subimg = img[y:n+y, x-n:x]
        quad = np.rot90(subimg, k=1)
    else: # quad=="D":
        subimg = img[y:n+y, x:n+x]
        quad = subimg

    if HAS_TORCH and not numpy_only:
        quad = torch.tensor(quad.copy()).view((1,1,n,n)).to("cpu")

    return quad

