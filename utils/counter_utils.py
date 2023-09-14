import numpy as np
import torch
import torchvision
from resonet.arches import CounterRn

"""
utils for pre-processing images before inputting them to the spot counter Resnet
Code copied from https://github.com/Isaac-Shuman/isashomod.git
"""


def process_image(img, cond_meth, useSqrt=True, lt=0, dev="cpu"):
    """

    Parameters
    ----------
    img: numpy array of shape (2527, 2463)
    cond_meth: conditioning method from this file
    useSqrt: whether to sqrt the counts
    lt: lower threshold (usually 0)
    dev: torch device to load tensor onto

    Returns
    -------
    processed tensor of shape (1,832,832)
    """
    if not img.dtype == np.float32:
        img = img.astype(np.float32)
    cond_img = torch.tensor(img).unsqueeze(0)
    cond_img = cond_img.to(dev)
    cond_img = cond_meth(cond_img)  # .squeeze()

    cond_img[cond_img < lt] = lt
    if useSqrt:
        cond_img = torch.sqrt(cond_img)
    return cond_img


def mx_gamma(dev=None, stride=3):
    """

    Parameters
    ----------
    dev: torch device
    factor: downsampling factor (should be 3 for Pilatus 6M and 5 for Eiger 16M)
    Returns
    -------
    Torch Compose object
    """
    mp = torch.nn.MaxPool2d(stride, stride=stride)
    if dev is not None:
        mp = mp.to(dev)
    tran = torchvision.transforms.Compose([
        mp,  # outputs image of size (842, 821)
        torchvision.transforms.CenterCrop(832)
    ])
    return tran


def load_count_model(model_path, model_arch):
    """

    Parameters
    ----------
    model_path: model state path
    model_arch: string of model arch

    Returns
    -------

    """
    num = int(model_arch.split("res")[1])
    model = CounterRn(num=num, two_fc_mode=False)
    model = load_model(model_path, model)
    model.eval()
    return model


def load_model(model_path, model):
    """

    Parameters
    ----------
    model_path: model state file
    model: model class instance

    Returns
    -------
    model
    """
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model
