
import numpy as np
import torch
import warnings
from resonet import arches


from resonet import params


def test_resnet50_gpu():
    main(50)
    main(50,num_out=2)
    main(50,num_out=2,num_geom=20)
    main(50,num_out=3,num_geom=20,nchan=3)


def test_resnet50_cpu():
    dev="cpu"
    main(50, dev=dev)
    main(50,num_out=2, dev=dev)
    main(50,num_out=2, dev=dev, num_geom=20)
    main(50,num_out=3, dev=dev, num_geom=20,nchan=3)


def test_resnet_wrapper():
    nout=2
    ngeom=6
    nchan=2
    bs=4
    dev="cuda:0"
    model = params.res50(nout=nout, ngeom=ngeom, nchan=nchan, dev=dev, pretrained=True)
    model2 = arches.RESNetAny(netnum=50,nout=nout, nchan=nchan, ngeom=ngeom, dev=dev, pretrained=True)
    assert str(model)==str(model2)
    print("ok")


def main(resnet_num, num_out=1, num_geom=5, dev="cuda:0", nchan=1):
    """
    :param resnet_num: resnet number (18,34,50,101,152)
    :param num_out:  number of output values per input
    :param num_geom:  length of geometry meta-data vector
    :param nchan:  number of image channels
    """
    bs = 4  # batch size

    image = np.random.random((4,nchan,512,512)).astype(np.float32)
    geom = np.random.random((4,num_geom)).astype(np.float32)

    image = torch.tensor(image).to(dev)
    geom = torch.tensor(geom).to(dev)

    model = params.ARCHES["res%d"%resnet_num](nout=num_out, ngeom=num_geom, dev=dev, nchan=nchan)
    # note, I had a weird warning on pytorch v 1.9.0,
    #  its been fixed, but I leave this catch here just in case..
    # https: // github.com / pytorch / pytorch / issues / 60053
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model(image)
        out2 = model(image, geom)
    assert out.shape == (bs, num_out)
    assert out2.shape == (bs, num_out)
    print("ok!")


if __name__=="__main__":
    print("test res50 GPU")
    test_resnet50_gpu()
    print("test res50 CPU")
    test_resnet50_cpu()
    print("test resnet wrapper")
    test_resnet_wrapper()
