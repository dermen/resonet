
from abc import abstractmethod

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torchvision.models import ResNet18_Weights

from resonet.utils import orientation


class OriQuatModel(torch.nn.Module):
    def __init__(self,img_sh, hidden_dim_resnet=128, trans_lays=6, res_layers=50, num_heads=8,
                 pretrained=True):
        super().__init__()
        assert hidden_dim_resnet % num_heads == 0
        
        # TODO: consider requiring even division by 32
        ydim, xdim=img_sh
        seq_len = ydim//32* xdim//32
        
        assert res_layers in {18,34,50}
        # strip out avg pool from resnet
        if res_layers==50:
            model = models.resnet.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            resnet_out = 2048

        elif res_layers==34:
            model = models.resnet.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            resnet_out=512
        elif res_layers==18:
            model = models.resnet.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            resnet_out=512
        else:
            raise NotImplementedError("only support 18,34,50 resnet layers")

        # overwrite first conv layer to accept 1-channel; images
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        hidden_dim_geom=15
        layers = list(model.children())
        self.mod = torch.nn.Sequential(*layers[:-2])

        # make linear layers
        self.lin1 = torch.nn.Linear(resnet_out,hidden_dim_resnet)
        self.lin2 = torch.nn.Linear(hidden_dim_resnet+hidden_dim_geom, 96)
        self.lin3 = torch.nn.Linear(96,4) # quaternion output
        
        # encode the position of the resnet features
        self.pos = torch.nn.Parameter(torch.randn(1,seq_len + 1, hidden_dim_resnet))

        # make the transofmer
        dim_ff= hidden_dim_resnet * 4 #
        dropout_rate = 0.1
        enc_lay = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim_resnet,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout_rate,
            activation='relu', # TODO: test also 'gelu'
            batch_first=True 
        )
        
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=enc_lay,
            num_layers=trans_lays
        )
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, hidden_dim_resnet))
        self.geom_mlp = torch.nn.Sequential(
            torch.nn.Linear(2,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,hidden_dim_geom))
        dist_ave = 300
        dist_sig = 115.47
        wave_ave = 1.125
        wave_sig = 0.101
        geom_ave =torch.tensor(np.array([dist_ave, wave_ave]).astype(np.float32))
        geom_sig =torch.tensor(np.array([dist_sig, wave_sig]).astype(np.float32))
        self.register_buffer('geom_ave', geom_ave)
        self.register_buffer('geom_sig', geom_sig)

        
    def forward(self,x, y):
        # x is the raw diffraction image
        # y is the pre-normalized geometry vector (distance, wavelength)
        y_norm = (y-self.geom_ave) / self.geom_sig
        
        bs = x.shape[0]
        x = self.mod(x).flatten(2)
        x = self.lin1(x.transpose(2,1))
        cls_tokens = self.cls_token.expand(bs, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x+self.pos
        x = self.transformer_encoder(x)
        x = x[:,0]
        geom_feat = self.geom_mlp(y_norm)
        x = torch.cat((x, geom_feat), dim=1)
        x = self.lin2(x)
        x = torch.nn.functional.relu(x)
        x = self.lin3(x)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x


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
