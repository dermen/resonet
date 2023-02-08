
import os
import re
import glob
import pandas
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import h5py

from PIL import Image


class PngDset(Dataset):

    def __init__(self, pngdir=None, propfile=None, quad="A", start=None, stop=None,
                 dev=None, invert_res=True):
        if pngdir is None:
            pngdir = "/global/cfs/cdirs/m3992/png/"
        if propfile is None:
            propfile = "/global/cfs/cdirs/m3992/png/num_reso_mos_B_icy1_icy2_cell_SGnum_pdbid_stolid.txt"
        if dev is None:
            dev = "cuda:0"

        self.fnames = glob.glob(os.path.join(pngdir, "*%s.png" % quad))
        assert self.fnames
        self.nums = [self.get_num(f) for f in self.fnames]
        self.img_sh = 546, 518
        self.props = ["reso"]

        self.prop = pandas.read_csv(
            propfile,
            delimiter=r"\s+",
            names=["num", "reso", "mos", "B", "icy1", "icy2", "cell1", \
                   "cell2", "cell3", "SGnum", "pdbid", "stolid"])

        if invert_res:
            self.prop.loc[:,"reso"] = 1/self.prop.reso

        self.labels = self.prop[["num", "reso"]]

        self.dev = dev  # pytorch device ID

        Ntotal = len(self.fnames)
        if start is None:
            start = 0
        if stop is None:
            stop = Ntotal
        assert start >= 0
        assert stop <= Ntotal
        assert stop > start
        self.start = start
        self.stop = stop

    @staticmethod
    def get_num(f):
        s = re.search("sim_[0-9]{5}", f)
        num = f[s.start(): s.end()].split("sim_")[1]
        return int(num)

    @property
    def dev(self):
        return self._dev

    @dev.setter
    def dev(self, val):
        self._dev = val

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, i):
        assert self.dev is not None, "Set the dev (torch device) property first!"

        img = Image.open(self.fnames[i+self.start])
        img_dat = np.reshape(img.getdata(), self.img_sh).astype(np.float32)

        num = self.nums[i+self.start]
        img_lab = self.labels.query("num==%d" % num).reso

        img_dat = torch.tensor(img_dat[:512,:512][None]).to(self.dev)
        img_lab = torch.tensor(img_lab.values).to(self.dev)
        return img_dat, img_lab


class H5SimDataDset(Dataset):

    def __init__(self, h5name, dev=None, labels="labels", images="images", start=None, stop=None,
                 label_sel=None, invert_labels=None):
        self.h5 = h5py.File(h5name, "r")
        self.images = self.h5[images]
        if label_sel is None:
            label_sel = [0]
        self.labels = self.h5[labels][:, label_sel]
        if invert_labels is None:
            invert_labels = [False]
        for i, inv in enumerate(invert_labels):
            if inv:
                self.labels[:,i] = 1/self.labels[:,i]
        self.dev = dev  # pytorch device ID
        if start is None:
            start = 0
        if stop is None:
            stop = self.images.shape[0]
        assert start >= 0
        assert stop <= self.images.shape[0]
        assert stop > start
        self.start = start
        self.stop = stop

    @property
    def dev(self):
        return self._dev

    @dev.setter
    def dev(self, val):
        self._dev = val

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, i):
        assert self.dev is not None
        img_dat, img_lab = self.images[i + self.start][None], self.labels[i + self.start]
        img_dat = torch.tensor(img_dat).to(self.dev)
        img_lab = torch.tensor(img_lab).to(self.dev)
        return img_dat, img_lab

    @property
    def nlab(self):
        return self.labels.shape[-1]


if __name__=="__main__":

    train_imgs = PngDset(start=2000, stop=9000)
    train_imgs_validate = PngDset(start=2000, stop=3000)
    test_imgs = PngDset(start=1000, stop=2000)

    train_tens = DataLoader(train_imgs, batch_size=16, shuffle=True)
    train_tens_validate = DataLoader(train_imgs_validate, batch_size=16, shuffle=True)
    test_tens = DataLoader(test_imgs, batch_size=16, shuffle=True)

    imgs, labs = next(iter(train_tens))
    print(imgs.shape, labs.shape)
