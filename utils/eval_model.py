
import sys
import numpy as np

from resonet.utils import maxbin

try:
    import torch
    from resonet import net
except ImportError:
    pass


HAS_TORCH = "torch" in sys.modules

MODEL_A = "/data/blstaff/xtal/mwilson_data/diff_AI/nety_ep40.nn"
#MODEL_B = "/data/blstaff/xtal/mwilson_data/diff_AI/nety_ep40.nn"
MODEL_B = "/global/cscratch1/sd/dermen/3p15_noMulti/trial.2/nety_ep20.nn"


def load_model(state_name):
    assert HAS_TORCH
    model = net.RESNet50(dev="cpu")
    state = torch.load(state_name, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model = model.to("cpu")
    model = model.eval()
    return model
    

#MASK = np.load("/data/blstaff/xtal/mwilson_data/mask_mwils.npy")

def raw_img_to_tens(raw_img, MASK):
    img = maxbin.get_quadA(maxbin.img2int(raw_img*MASK))
    img = img.astype(np.float32)[:512,:512]
    if HAS_TORCH:
        img = torch.tensor(img).view((1,1,512,512)).to("cpu")
    return img


def raw_img_to_tens_pil(raw_img, MASK, xy=None):
    ysl, xsl = maxbin.get_slice_pil(xy)
    # or else pad img if shape is not 1024x1024
    img = maxbin.img2int_pil(raw_img[ysl, xsl]*MASK[ysl,xsl])
    img = maxbin.get_quadA_pil(img).astype(np.float32)
    if HAS_TORCH:
        img = torch.tensor(img).view((1,1,512,512)).to("cpu")
    return img


def res_img_to_tens_pil(raw_img, MASK, xy=None):
    ysl, xsl = maxbin.get_slice_pil(xy)
    # or else pad img if shape is not 1024x1024
    img = maxbin.img2int_pil(raw_img[ysl, xsl]*MASK[ysl,xsl])
    img = maxbin.get_quadA_pil(img).astype(np.float32)
    if HAS_TORCH:
        img = torch.tensor(img).view((1,1,512,512)).to("cpu")
    return img
