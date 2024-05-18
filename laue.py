import sys
from ptunet.unet.unet_model import UNet
import torch


class SUNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.unet = UNet(*args, **kwargs)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        nbatch, _, nslow, nfast = x.shape
        nclass = self.unet.n_classes
        x = self.unet(x).view(nbatch, nclass, -1)
        x = self.softmax(x).view(nbatch, nclass, nslow, nfast)
        return x

