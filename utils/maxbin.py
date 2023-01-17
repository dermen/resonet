# coding: utf-8
from PIL import Image
import numpy as np
import dxtbx

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


def img2int(raw_eiger):
    img = bin_ndarray(raw_eiger[:4360, :4148], SHAPE_MAXFILT, 'max')
    img[ img < 0] = 0
    img = np.sqrt(img)
    img[ img > IMAX] = IMAX 
    img = img.astype(np.uint8)
    return img[:SHAPE_CROP[0], :SHAPE_CROP[1]]


def img2int_pil(raw_pil):
    assert raw_pil.shape==(2048,2048)
    img = bin_ndarray(raw_pil, (1024,1024), 'max')
    img[ img < 0] = 0
    img = np.sqrt(img)
    img[ img > IMAX] = IMAX 
    img = img.astype(np.uint8)
    return img


def get_quadA(img):
    quadA = np.rot90(np.rot90(img))[-SHAPE_CROP[0]//2:,-SHAPE_CROP[1]//2:]
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


if __name__=="__main__":
    # reference image from cbf2int, created using the following 2 commands:
    # ~blstaff/generation_scripts/cbf2int  -maxpool 4 --sqrt --bits 8  -pgm  /data/lyubimov/software_tests/eiger2_data/10242021/B2/B2_1_00573.cbf -output test2.pgm
    # convert -crop 1030x1028+0+0 test2.pgm  +repage test3.png
    img0 = np.array(Image.open("/home/blstaff/generation_scripts/test3.png").getdata()).reshape(SHAPE_CROP)
    
    # reference quadrant image created using:
    # convert -flip -flop -crop 515x514+515+514 test3.png  +repage  test3_A.png
    img0_A = np.array(Image.open("/home/blstaff/generation_scripts/test3_A.png").getdata()).reshape(SHAPE_CROP[0]//2, SHAPE_CROP[1]//2)

    # load raw image that made reference image
    loader = dxtbx.load("/data/lyubimov/software_tests/eiger2_data/10242021/B2/B2_1_00573.cbf")
    img = loader.get_raw_data().as_numpy_array()

    # test full image conversion using max filter, compare with cbf2int
    img2 = img2int(img)
    assert np.allclose(img0, img2)
    # test quadrant selection, compare with convert flip/flop + crop
    img2_A = get_quadA(img2)
    assert np.allclose(img0_A, img2_A)
    print("ok!")

