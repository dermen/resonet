
from abc import abstractmethod

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet34, resnet101, resnet152

class RESNetBase(nn.Module):

    def _set_blocks(self):
        self.resnet.conv1 = nn.Conv2d(self.nout, 64,
                                      kernel_size=7, stride=2, padding=3, bias=False,
                                      device=self.dev)
        self.DROP = nn.Dropout(p=0.5)
        #if self.dropout:
        #    dropout = nn.Dropout(p=0.5)
        #    self.fc1 = dropout(nn.Linear(1000, 100, device=self.dev))
        #    self.fc2 = dropout(nn.Linear(100, self.nout, device=self.dev))
        #else:
        self.fc1 = nn.Linear(1000, 100, device=self.dev)
        self.fc2 = nn.Linear(100, self.nout, device=self.dev)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        if self.dropout:
            x = self.DROP(F.relu(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.binary:
            x = self.Sigmoid(x)
        return x

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


class RESNet50BC(RESNetBase):
    def __init__(self, dev=None, nout=1):
        super().__init__()
        if dev is None:
            self.dev = "cuda:0"
        self.nout = nout
        self.resnet = resnet50().to(self.dev)
        self._set_blocks()
        self.binary = True


class RESNet18BC(RESNetBase):
    def __init__(self, dev=None, nout=1):
        super().__init__()
        if dev is None:
            self.dev = "cuda:0"
        self.nout = nout
        self.resnet = resnet18().to(self.dev)
        self._set_blocks()
        self.binary = True


class RESNetAny(RESNetBase):

    def __init__(self, num=18, dev=None, device_id=0, nout=1, dropout=False):
        super().__init__()
        self.dropout = dropout
        if dev is None:
            self.dev = "cuda:%d" % device_id
        else:
            self.dev = dev
        self.nout = nout
        if num==18:
            self.resnet = resnet18().to(self.dev)
        elif num==50:
            self.resnet = resnet18().to(self.dev)
        elif num==34:
            self.resnet = resnet34().to(self.dev)
        elif num==101:
            self.resnet=resnet101().to(self.dev)
        else:
            self.resnet = resnet152().to(self.dev)

        self.binary = False
        self._set_blocks()


class RESNet18(RESNetBase):
    def __init__(self, dev=None, device_id=0, nout=1, dropout=False):
        super().__init__()
        self.dropout = dropout
        if dev is None:
            self.dev = "cuda:%d" % device_id
        else:
            self.dev = dev
        self.nout = nout
        self.resnet = resnet18().to(self.dev)
        self.binary = False
        self._set_blocks()


class RESNet50(RESNetBase):
    def __init__(self, dev=None, device_id=0, nout=1, dropout=False):
        super().__init__()
        self.dropout = dropout
        if dev is None:
            self.dev = "cuda:%d" % device_id
        else:
            self.dev = dev
        self.nout = nout
        self.resnet = resnet50().to(self.dev)
        self.binary = False
        self._set_blocks()


class RESNet34(RESNetBase):
    def __init__(self, dev=None, device_id=0, nout=1, dropout=False):
        super().__init__()
        self.dropout = dropout
        if dev is None:
            self.dev = "cuda:%d" % device_id
        else:
            self.dev = dev
        self.nout = nout
        self.resnet = resnet34().to(self.dev)
        self.binary = False
        self._set_blocks()


class RESNet101(RESNetBase):
    def __init__(self, dev=None, device_id=0, nout=1, dropout=False):
        super().__init__()
        self.dropout = dropout
        if dev is None:
            self.dev = "cuda:%d" % device_id
        else:
            self.dev = dev
        self.nout = nout
        self.resnet = resnet101().to(self.dev)
        self.binary = False
        self._set_blocks()


class RESNet152(RESNetBase):
    def __init__(self, dev=None, device_id=0, nout=1, dropout=False):
        super().__init__()
        self.dropout = dropout
        if dev is None:
            self.dev = "cuda:%d" % device_id
        else:
            self.dev = dev
        self.nout = nout
        self.resnet = resnet152().to(self.dev)
        self.binary = False
        self._set_blocks()


class LeNet(nn.Module):
    def __init__(self, dev=None, nout=1, dropout=False):
        super().__init__()
        if dev is None:
            self.dev = "cuda:0"
        else:
            self.dev = dev
        self.dropout=dropout
        self.DROP = nn.Dropout(p=0.5)
        self.nout = nout
        self.conv1 = nn.Conv2d(1, 6, 3, device=self.dev)
        self.conv2 = nn.Conv2d(6, 16, 3, device=self.dev)
        self.conv2_bn = nn.BatchNorm2d(16, device=self.dev)
        self.conv3 = nn.Conv2d(16, 32, 3, device=self.dev)
        self.conv3_bn = nn.BatchNorm2d(32, device=self.dev)

        self.fc1 = nn.Linear(32 * 62 * 62, 1000, device=self.dev)
        self.fc2 = nn.Linear(1000, 100, device=self.dev)
        self.fc3 = nn.Linear(100, self.nout, device=self.dev)
        self.relu = nn.ReLU()

    def forward(self, x):
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

        x = self.fc3(x)
        return x

