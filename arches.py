
from abc import abstractmethod

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models

from resonet.utils import orientation


class RESNetBase(nn.Module):

    def _set_blocks(self):
        padding = int(round(self.kernel_size/2)) - 1
        self.resnet.conv1 = nn.Conv2d(self.nchan, 64,
                                      kernel_size=self.kernel_size, stride=2, padding=padding, bias=False,
                                      device=self.dev)
        self.DROP = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1000, self.num_fc, device=self.dev)
        self.fc2 = nn.Linear(self.num_fc, self.nout, device=self.dev)
        # optional linear model including geometry
        self.fc2_geom = nn.Linear(self.num_fc+self.ngeom, self.nout, device=self.dev)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, y=None):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        if self.dropout:
            x = self.DROP(F.relu(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.binary:
            x = self.Sigmoid(x)
        if y is not None:
            if y.shape[-1] == 4:  # NEWWAY provide downsampling factor directly
                detdist, pixsize, wavelen, fact = y.T

            elif y.shape[-1] == 5:  # OLDWAY, geom is xdim+ydim
                detdist, pixsize, wavelen, fact, _ = y.T
                is_pilatus = fact==2463
                # convert xdim to a downsampling term:
                fact[is_pilatus] = 2
                fact[~is_pilatus] = 4
            else:
                raise ValueError("unsupported y shape")

            theta = torch.arctan(((fact * pixsize / detdist) * x.T).T) * 0.5
            stheta = torch.sin(theta)
            # NOTE this is for 1/reso
            x = ((2/wavelen)*stheta.T).T
        if self.ori_mode:
            x = orientation.gs_mapping(x)
        return x

    @property
    @abstractmethod
    def num_fc(self):
        """number of output channels"""
        return self._num_fc

    @num_fc.setter
    @abstractmethod
    def num_fc(self, val):
        if val >= 1000 or val < 10:
            raise ValueError("num_fc should be < 1000 and >= 10")
        self._num_fc = val


    @property
    @abstractmethod
    def nout(self):
        """number of output channels"""
        return self._nout

    @nout.setter
    @abstractmethod
    def nout(self, val):
        self._nout = val

    @property
    @abstractmethod
    def dev(self):
        """pytorch device id e.g. `cuda:0`"""
        return self._dev

    @dev.setter
    @abstractmethod
    def dev(self, val):
        self._dev = val

    @property
    @abstractmethod
    def ori_mode(self):
        return self._ori_mode

    @ori_mode.setter
    @abstractmethod
    def ori_mode(self, val):
        self._ori_mode = val

    @property
    @abstractmethod
    def binary(self):
        return self._binary

    @binary.setter
    @abstractmethod
    def binary(self, val):
        self._binary = val

    @property
    @abstractmethod
    def dropout(self):
        return self._dropout

    @dropout.setter
    @abstractmethod
    def dropout(self, val):
        self._dropout = val

    @property
    @abstractmethod
    def ngeom(self):
        return self._ngeom

    @ngeom.setter
    @abstractmethod
    def ngeom(self, val):
        self._ngeom = val

    @property
    @abstractmethod
    def nchan(self):
        return self._nchan

    @nchan.setter
    @abstractmethod
    def nchan(self, val):
        self._nchan = val

    @property
    @abstractmethod
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    @abstractmethod
    def kernel_size(self, val):
        self._kernel_size = val


class RESNetAny(RESNetBase):
    # not used anywhere yet...

    def __init__(self, netnum, dev=None, device_id=0, nout=1, dropout=False, ngeom=5, nchan=1,
                 weights=None, kernel_size=7, num_fc=100):
        """

        :param netnum: resnet number (18,34,50,101,152)
        :param dev: pytorch device
        :param device_id: gpu id(only matters if dev is None)
        :param nout: number of output channels
        :param dropout: whether to use dropout layer
        :param ngeom: length of geometry meta-data vector
        :param nchan: number of channels in input image (e.g. RGB images have 3 channels)
        :param weights: whether to use the pretrained resnet models, and specify weights
        :param kernel_size: the size of the conv1 kernel in the resnet
        :param num_fc: the number of output channels of fc1 whose inputs are the 1000 resnet outputs
            this number should be < 1000 and >= 10
        """
        super().__init__()
        self.dropout = dropout
        self.nchan = nchan
        self.kernel_size = kernel_size
        self.ngeom= ngeom
        if dev is None:
            self.dev = "cuda:%d" % device_id
        else:
            self.dev = dev
        self.nout = nout
        self.num_fc = num_fc
        model = getattr(models, "resnet%d" % netnum)
        try:
            self.resnet = model(weights=weights).to(self.dev)
        except TypeError:
            self.resnet = model().to(self.dev)

        self.binary = False
        self.ori_mode = False
        self._set_blocks()


class LeNet(nn.Module):
    def __init__(self, dev=None, nout=1, dropout=False, ngeom=5, nchan=1, kernel_size=None):
        """

        :param dev: pytorch device
        :param nout: number of output channels
        :param dropout: whether to use a dropout layer
        :param ngeom: length of meta-data vector
        :param nchan: number of input image channels
        :param kernel_size: Unused
        """
        super().__init__()
        self.ngeom=ngeom
        if dev is None:
            self.dev = "cuda:0"
        else:
            self.dev = dev
        self.dropout=dropout
        self.DROP = nn.Dropout(p=0.5)
        self.nout = nout
        self.nchan = nchan
        self.conv1 = nn.Conv2d(self.nchan, 6, 3, device=self.dev)
        self.conv2 = nn.Conv2d(6, 16, 3, device=self.dev)
        self.conv2_bn = nn.BatchNorm2d(16, device=self.dev)
        self.conv3 = nn.Conv2d(16, 32, 3, device=self.dev)
        self.conv3_bn = nn.BatchNorm2d(32, device=self.dev)

        self.fc1 = nn.Linear(32 * 62 * 62, 1000, device=self.dev)
        self.fc2 = nn.Linear(1000, 100, device=self.dev)
        self.fc3 = nn.Linear(100, self.nout, device=self.dev)
        self.fc3_geom = nn.Linear(100+ngeom, self.nout, device=self.dev)
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = torch.flatten(x, 1)

        if self.dropout:
            x = F.relu(self.DROP(self.fc1(x)))
            x = F.relu(self.DROP(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

        if y is not None:
            x = torch.concat((x, y), dim=1)
            x = self.fc3_geom(x)
        else:
            x = self.fc3(x)

        return x


class CounterRn(nn.Module):
    """
    Spot Counter resnet
    copied from https://github.com/Isaac-Shuman/isashomod.git
    """
    def __init__(self, num=18, two_fc_mode=False):
        super().__init__()

        self.two_fc_mode = two_fc_mode

        if num == 18:
            self.res = models.resnet18()
        if num == 34:
            self.res = models.resnet34()
        if num == 50:
            self.res = models.resnet50()

        self.res.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc_1000_to_100 = nn.Linear(1000, 100)
        self.fc_100_to_1 = nn.Linear(100, 1)
        self.fc_1000_to_1 = nn.Linear(1000, 1)

    def forward(self, x):

        x = self.res(x)
        x = F.relu(x)

        if self.two_fc_mode:
            x = self.fc_1000_to_100(x)
            x = F.relu(x)
            x = self.fc_100_to_1(x)
        else:
            x = self.fc_1000_to_1(x)

        return x
