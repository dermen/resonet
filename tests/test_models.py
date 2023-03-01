# run with
# pytest -q tests/test_models.py

import numpy as np
import torch
import warnings
from resonet import arches
from resonet import params

class TestModels:

    def test_resnet50_gpu(self):
        self.main(50)
        self.main(50,num_out=2)
        self.main(50,num_out=2,num_geom=20)
        self.main(50,num_out=3,num_geom=20,nchan=3)
        self.main(50,num_out=3,num_geom=20,nchan=3, weight = "IMAGENET1K_V1")

    def test_resnet50_cpu(self):
        dev="cpu"
        self.main(50, dev=dev)
        self.main(50,num_out=2, dev=dev)
        self.main(50,num_out=2, dev=dev, num_geom=20)
        self.main(50,num_out=3, dev=dev, num_geom=20,nchan=3)
        self.main(50,num_out=3,num_geom=20,nchan=3, weight = "IMAGENET1K_V1")

    def test_resnet18_gpu(self):
        self.main(18)
        self.main(18,num_out=2)
        self.main(18,num_out=2,num_geom=20)
        self.main(18,num_out=3,num_geom=20,nchan=3)
        self.main(18,num_out=3,num_geom=20,nchan=3, weight = "IMAGENET1K_V1")


    def test_resnet18_cpu(self):
        dev="cpu"
        self.main(18, dev=dev)
        self.main(18,num_out=2, dev=dev)
        self.main(18,num_out=2, dev=dev, num_geom=20)
        self.main(18,num_out=3, dev=dev, num_geom=20,nchan=3)
        self.main(18,num_out=3,num_geom=20,nchan=3, weight = "IMAGENET1K_V1")


    def test_resnet_wrapper(self):
        nout=2
        ngeom=6
        nchan=2
        bs=4
        dev="cuda:0"
        model = params.res50(nout=nout, ngeom=ngeom, nchan=nchan, dev=dev)
        model2 = arches.RESNetAny(netnum=50,nout=nout, nchan=nchan, ngeom=ngeom, dev=dev)
        print(model)
        assert str(model)==str(model2)

    def main(self, resnet_num, num_out=1, num_geom=5, dev="cuda:0", nchan=1, weight = None):
        """
        :param resnet_num: resnet number (18,34,50,101,152)
        :param num_out:  number of output values per input
        :param num_geom:  length of geometry meta-data vector
        :param nchan:  number of image channels
        :param weight: pretrained weights
        """
        bs = 4  # batch size

        image = np.random.random((4,nchan,512,512)).astype(np.float32)
        geom = np.random.random((4,num_geom)).astype(np.float32)

        image = torch.tensor(image).to(dev)
        geom = torch.tensor(geom).to(dev)

        model = params.ARCHES["res%d"%resnet_num](nout=num_out, ngeom=num_geom, dev=dev, nchan=nchan,weights = weight)
        # note, I had a weird warning on pytorch v 1.9.0,
        #  its been fixed, but I leave this catch here just in case..
        # https: // github.com / pytorch / pytorch / issues / 60053
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model(image)
            out2 = model(image, geom)
        assert out.shape == (bs, num_out)
        assert out2.shape == (bs, num_out)


if __name__=="__main__":
    t = TestModels()
    print("test res50 GPU")
    t.test_resnet50_gpu()
    print("test res50 CPU")
    t.test_resnet50_cpu()
    print("test res18 GPU")
    t.test_resnet18_gpu()
    print("test res18 CPU")
    t.test_resnet18_cpu()
    print("test resnet wrapper")
    t.test_resnet_wrapper()
