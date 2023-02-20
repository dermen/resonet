
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
                 dev=None, invert_res=True,transform = None):
        if pngdir is None:
            pngdir = "/global/cfs/cdirs/m3992/png/"
        if propfile is None:
            propfile = "/global/cfs/cdirs/m3992/png/num_reso_mos_B_icy1_icy2_cell_SGnum_pdbid_stolid.txt"
        if dev is None:
            dev = "cuda:0"
        
        self.transform = transform

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
        # Apply image transform here
        if self.transform:
            img_dat = self.transform(img_dat)
        return img_dat, img_lab


class H5SimDataDset(Dataset):

    def __init__(self, h5name, dev=None, labels="labels", images="images",
                 start=None, stop=None, label_sel=None, use_geom=False):
        """

        :param h5name: hdf5 master file written by resonet/scripts/merge_h5s.py
        :param dev: pytorch device
        :param labels: path to labels dataset
        :param images: path to images dataset
        :param start: dataset index to begin
        :param stop: dataset index to stop
        :param label_sel: optional list of labels to select. E.g. if labels is a dataset whose
            shape is (Nsamples,5), e.g. 5 labels per sample, and you want to serve up labels 0 and 2, then
            you should set label_sel=[0,2]
            Default is [0], just use the first label
        :param use_geom: if the `geom` dataset exists, then use each iter should return 3-tuple (labels, images, geom)
            Otherwise, each iter returns 2-tuple (images,labels)
            The geom tensor can be used as a secondary input to certain models
        """
        if label_sel is None:
            label_sel = [0]
        self.nlab = len(label_sel)
        self.label_sel = label_sel
        self.labels_name = labels  #
        self.images_name = images
        self.h5name = h5name

        self.has_geom = False  # if geometry is present in master file, it can be used as model input
        # open to get length quickly!
        with h5py.File(h5name, "r") as h:
            self.num_images = h[self.images_name].shape[0]
            self.has_geom = "geom" in list(h.keys())
        if use_geom and not self.has_geom:
            raise ValueError("Cannot use geometry if it is not present in the master files. requires `geom` dataset")
        self.use_geom = use_geom and self.has_geom

        self.dev = dev  # pytorch device ID
        if start is None:
            start = 0
        if stop is None:
            stop = self.num_images
        assert start >= 0
        assert stop <= self.num_images
        assert stop > start
        self.start = start
        self.stop = stop

        self.h5 = None  # handle for hdf5 file
        self.images = None  # hdf5 dataset
        self.labels = None  # hdf5 dataset
        self.geom = None  # hdf5 dataset

    def open(self):
        self.h5 = h5py.File(self.h5name, "r")
        self.images = self.h5[self.images_name]
        if self.images.dtype!=np.float32:
            raise ValueError("Images should be type float32!")
        self.labels = self.h5[self.labels_name][:, self.label_sel]
        if self.labels.dtype==np.float64:
            self.labels = self.labels.astype(np.float32)
        if self.use_geom:
            self.geom = self.h5["geom"][()]
            if self.geom.dtype==np.float64:
                self.geom = self.geom.astype(np.float32)

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
        if self.images is None:
            self.open()
        img_dat, img_lab = self.images[i + self.start][None], self.labels[i + self.start]
        img_dat = torch.tensor(img_dat).to(self.dev)
        img_lab = torch.tensor(img_lab).to(self.dev)
        if self.use_geom:
            geom_inputs = self.geom[i+self.start]
            geom_inputs = torch.tensor(geom_inputs).to(self.dev)
            return img_dat, img_lab, geom_inputs
        else:
            return img_dat, img_lab

    def nlab(self):
        return self.nlab


class H5SimDataMPI(H5SimDataDset):

    def __init__(self, mpi_comm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.com = mpi_comm

    def __getitem__(self, i):
        assert self.dev is not None
        img_dat, img_lab = self.images[i + self.start][None], self.labels[i + self.start]
        return img_dat, img_lab
        #img_dat = torch.tensor(img_dat).to(self.dev)
        #img_lab = torch.tensor(img_lab).to(self.dev)
        #return img_dat, img_lab


class H5Loader:
    def __init__(self, dset, comm, batch_size=8, shuffle=True):
        self.shuffle = shuffle
        self.dset = dset
        self.comm = comm
        self.bs = batch_size
        self.i_batch = 0  # batch counter
        self.samp_per_batch = None
        self.batch_data_holder = None
        self.batch_label_holder = None
        self._set_batches()

    def __iter__(self):
        return self

    def __next__(self):

        self.i_batch += 1
        if self.i_batch < len(self.samp_per_batch):
            return self.get_batch()
        else:
            self._set_batches()
            self.i_batch = 0
            raise StopIteration

    def _set_batches(self):
        nbatch = int(len(self.dset) / self.bs)

        if self.comm.rank == 0:
            batch_order = np.random.permutation(len(self.dset))
            self.samp_per_batch = np.array_split(batch_order, nbatch)
        self.samp_per_batch = self.comm.bcast(self.samp_per_batch)
        max_size = max([len(x) for x in self.samp_per_batch])
        self.batch_data_holder = np.zeros((max_size, 1, 512, 512))
        self.batch_label_holder = np.zeros((max_size, 1))

    def get_batch(self):
        assert self.i_batch < len(self.samp_per_batch)
        nsamp = len(self.samp_per_batch[self.i_batch])
        for ii, i in enumerate(self.samp_per_batch[self.i_batch]):
            if i % self.comm.size != self.comm.rank:
                continue
            data, label = self.dset[i]
            self.batch_data_holder[ii] = data
            self.batch_label_holder[ii] = label
        self.batch_data_holder = self._reduce_bcast(self.batch_data_holder)
        self.batch_label_holder = self._reduce_bcast(self.batch_label_holder)
        dat = torch.tensor(self.batch_data_holder[:nsamp])
        lab = torch.tensor(self.batch_label_holder[:nsamp])
        return dat, lab

    def _reduce_bcast(self, arr):
        return self.comm.bcast(self.comm.reduce(arr))


if __name__=="__main__":

    train_imgs = PngDset(start=2000, stop=9000)
    train_imgs_validate = PngDset(start=2000, stop=3000)
    test_imgs = PngDset(start=1000, stop=2000)

    train_tens = DataLoader(train_imgs, batch_size=16, shuffle=True)
    train_tens_validate = DataLoader(train_imgs_validate, batch_size=16, shuffle=True)
    test_tens = DataLoader(test_imgs, batch_size=16, shuffle=True)

    imgs, labs = next(iter(train_tens))
    print(imgs.shape, labs.shape)
