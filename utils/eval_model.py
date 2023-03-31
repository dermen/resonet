
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

