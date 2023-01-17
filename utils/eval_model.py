
import sys
import numpy as np

from sx.diffraction_ai import maxbin

try:
    import torch
    import resonet
except ImportError:
    pass


HAS_TORCH = "torch" in sys.modules

MODEL_A = "/data/blstaff/xtal/mwilson_data/diff_AI/nety_ep40.nn"


def load_model(state_name):
    assert HAS_TORCH
    assert "resonet" in sys.modules
    model = resonet.net.RESNet50()
    state = torch.load(state_name)
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
 
