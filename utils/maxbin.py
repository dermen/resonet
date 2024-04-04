# coding: utf-8
from PIL import Image
import numpy as np
import torch
from torch.nn.functional import pad as PAD

SHAPE_MAXFILT = 1090, 1037
SHAPE_CROP = 1028, 1030
IMAX = 127  #np.iinfo(np.uint8).max


def bin_ndarray(ndarray, new_shape, how='max'):
    assert how in ['max', 'mean']
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    if how=='max':
        for i in range(len(new_shape)):
            ndarray = ndarray.max(-1*(i+1))
    elif how=='mean':
        for i in range(len(new_shape)):
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray


def convert_res_img(res_img):
    img = bin_ndarray(res_img[:4360, :4148], SHAPE_MAXFILT, 'mean')
    return img[:SHAPE_CROP[0], :SHAPE_CROP[1]]


def img2int_mar(raw_mar, howbin='max'):
    assert raw_mar.shape==(4096,4096)
    img = bin_ndarray(raw_mar, (1024,1024), howbin)
    img[ img < 0] = 0
    img = np.sqrt(img)
    img[ img > IMAX] = IMAX
    img = img.astype(np.uint8)
    return img

def img2int(raw_eiger, howbin='max'):
    img = bin_ndarray(raw_eiger[:4360, :4148], SHAPE_MAXFILT, howbin)
    img[ img < 0] = 0
    img = np.sqrt(img)
    img[ img > IMAX] = IMAX 
    img = img.astype(np.uint8)
    return img[:SHAPE_CROP[0], :SHAPE_CROP[1]]


def img2int_pil(raw_pil, howbin='max'):
    assert raw_pil.shape==(2048,2048)
    img = bin_ndarray(raw_pil, (1024,1024), howbin)
    img[ img < 0] = 0
    img = np.sqrt(img)
    img[ img > IMAX] = IMAX 
    img = img.astype(np.uint8)
    return img


def get_quadA(img):
    quadA = np.rot90(np.rot90(img))[-SHAPE_CROP[0]//2:,-SHAPE_CROP[1]//2:]
    return quadA

def get_quadA_mar(img):
    quadA = np.rot90(np.rot90(img))[-512:, -512:]
    return quadA


def get_quadA_pil(img):
    quadA = np.rot90(np.rot90(img))[-512:,-512:]
    return quadA


def get_slice_pil(xy=None):
    if xy is None:
        x,y = 1231, 1263
    else:
        assert len(xy)==2
        x,y = xy
        assert isinstance(x,int)
        assert isinstance(y,int)
    ysl = slice(y-1024,y+1024,1) 
    xsl = slice(x-1024,x+1024,1)
    return ysl, xsl


def downsample_tensor(tens, factor, maxpool):
    """

    Parameters
    ----------
    tens : torch.Tensor, 2-dimensional
    factor: down sampling factor
    maxpool: instance of torch.nn.MaxPool2D

    Returns
    -------
    downsampled tensor
    """
    ydim, xdim = tens.shape
    while xdim % factor:
        xdim += 1
    while ydim % factor:
        ydim += 1
    ypad = ydim - tens.shape[0]
    xpad = xdim - tens.shape[1]

    tens = PAD(PAD(tens, (0, xpad), value=0).T, (0, ypad), value=0).T
    tens = maxpool(tens[None])[0]
    return tens


def maximg_downsample(img, factor=2, maxpool=None, dev="cpu",
                      leave_on_gpu=False, convert_to_f32=False):
    """

    Parameters
    ----------
    img: np.ndarray
    factor: int factor
    maxpool: maxpool method
    dev: torch device

    Returns
    -------

    """
    ydim, xdim = img.shape
    while xdim % factor:
        xdim += 1
    while ydim % factor:
        ydim += 1
    ypad = ydim - img.shape[0]
    xpad = xdim - img.shape[1]
    if maxpool is not None:
        if convert_to_f32:
            img = img.astype(np.float32)
        padt = torch.tensor(img).to(dev) 
        padt = PAD(PAD(padt, (0, xpad), value=0).T, (0, ypad), value=0).T
        maximg = maxpool(padt[None])[0]
        if not leave_on_gpu:
            try:
                maximg = maximg.numpy()
            except TypeError:
                maximg = maximg.cpu().numpy()
    else:
        yzeros = np.zeros((ypad, img.shape[1]))
        padimg = np.concatenate((img, yzeros), axis=0)
        xzeros = np.zeros((ydim, xpad))
        padimg = np.concatenate((padimg , xzeros), axis=1)
        maximg = bin_ndarray(padimg, (int(ydim / factor), int(xdim / factor)), 'max')
    return maximg

