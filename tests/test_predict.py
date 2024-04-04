import numpy as np
from resonet.utils.predict_fabio import ImagePredictFabio
import os

def _test_predict_fabio(i):
    shapes = [(1920,1920), (3840,3840), (2527, 2463), (4096, 4096)]
    strides = 1,2,2,4
    use_modern_reso= i%2==0
    np.random.seed(0)
    img = np.random.random(shapes[i]).astype(np.float32)
    quads = [-2], [-1], [0,1], [0,1,2,3]
    ice_mask = i%2==0
    main_fabio(img, use_modern_reso=use_modern_reso,ds_stride=strides[i] , quads=quads[i], use_ice_mask=ice_mask)


def main_fabio(img, use_modern_reso=True, B_to_d=None, dev="cpu", mask=None, 
        cent=None, ds_stride=None, quads=[0], distance=100, pixsize=0.1, wavelen=1, gain=1,
        use_ice_mask=True):
    reso_model = "_pytest_resolution.nn"
    if not os.path.exists(reso_model):
        os.system("wget https://smb.slac.stanford.edu/~resonet/resolution.nn -O %s" % reso_model)

    P = ImagePredictFabio(
        reso_model=reso_model,
        multi_model=None,
        ice_model=None,
        counts_model=None,
        reso_arch="res50",
        multi_arch=None,
        ice_arch=None,
        counts_arch=None,
        dev=dev,
        use_modern_reso=use_modern_reso,
        B_to_d=B_to_d,
        )

    P.cent = cent
    P.quads = quads
    P.ds_stride = ds_stride
    P.gain = gain
    cent = None
    if use_ice_mask:
        cent = [x/2. for x in img.shape]
    P.load_image_from_file_or_array(detdist=distance,
        pixsize=pixsize, wavelen=wavelen, raw_image=img, beam_center=cent, 
        use_ice_mask=use_ice_mask)
    d = P.detect_resolution()
    d2 = P.detect_resolution(use_min=False)


def test_predict_fabio0():
    _test_predict_fabio(0)
def test_predict_fabio1():
    _test_predict_fabio(1)
def test_predict_fabio2():
    _test_predict_fabio(2)
def test_predict_fabio3():
    _test_predict_fabio(3)

