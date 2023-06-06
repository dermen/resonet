

import numpy as np
import torch

from resonet.utils.maxbin import maximg_downsample
from resonet.utils import eval_model

def _test_pil2(shape=None, factor=2, use_maxpool=False, dev="cpu", leave_on_gpu=False,
               convert_to_f32=False, camera="pilatus"):

    if shape is None:
        shape = 6,6
    img = np.arange(shape[0]*shape[1]).reshape(shape).astype(np.float32)

    print("testing maxpool algorithm in general")
    maxpool = torch.nn.MaxPool2d(factor,factor)
    maximg = maximg_downsample(img, factor=factor, maxpool=None, dev=dev,
                                 leave_on_gpu=leave_on_gpu, convert_to_f32=convert_to_f32)
    maximg2 = maximg_downsample(img, factor=factor, maxpool=maxpool, dev=dev,
                               leave_on_gpu=leave_on_gpu, convert_to_f32=convert_to_f32)
    maximg_test = [img[j,i] for j in range(factor-1,shape[0],factor) for i in range(factor-1, shape[0],factor)]
    maximg_test = np.reshape(maximg_test, maximg.shape)
    assert np.allclose(maximg_test, maximg)
    assert np.allclose(maximg2, maximg)
    print("OK!")

    if camera == "eiger":
        shape = 4371, 4150
        factor = 4
    else:
        shape = 2527, 2463
        factor = 2

    center = shape[1]/2., shape[0]/2.
    img = np.random.randint(0, 255**2, shape).astype(np.float32)
    img += np.random.random(shape)
    mask = np.ones_like(img).astype(bool)

    for mp in [None, maxpool]:
        print("Testing maxpool %s" % ("torch" if mp is not None else "bin_ndarray"))
        for q in ["A"]: #["A","B", "C", "D"]:
            quad = eval_model.raw_img_to_tens_pil2(img, mask, numpy_only=False,
                    cent=center, IMAX=None,
                    adu_per_photon=1, quad=q,
                    ds_fact=factor, sqrt=True, maxpool=mp, dev=dev, leave_on_gpu=False,
                    convert_to_f32=convert_to_f32)

            quad2 = eval_model.raw_img_to_tens_pil3(img, mask, numpy_only=False,
                                                   cent=center, IMAX=None,
                                                   adu_per_photon=1, quad=q,
                                                   ds_fact=factor, sqrt=True, maxpool=mp, dev=dev, leave_on_gpu=False,
                                                   convert_to_f32=convert_to_f32)
            if not  np.allclose(quad, quad2):
                from IPython import embed;embed()
            #assert np.allclose(quad, quad2)
        print("OK")


if __name__=="__main__":
    _test_pil2((100,100), factor=2, camera="pilatus")
    # TODO: determine if we actually care about making this following test pass...
    _test_pil2((200,200), factor=4, camera="eiger")
