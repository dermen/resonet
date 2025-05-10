
import os
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py


class CompressDset(Dataset):

    def __init__(self, h5name, maximgs=None):
        self.h5name = h5name
        self.h5 = None
        self.images = self.labels = None
        self.maximgs = maximgs

    def _open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5name, "r")

    def __len__(self):
        self._open()
        if self.maximgs is not None:
            assert self.maximgs <= self.h5["images"].shape[0]
            return self.maximgs
        else:
            return self.h5['images'].shape[0]

    def __getitem__(self, idx):
        self._open()
        img = torch.tensor(self.h5['images'][idx][None])
        lab = torch.tensor(self.h5['peak_segments'][idx])
        img = img.float()
        lab = lab.float()
        return img, lab


class H5SimDataDset(Dataset):

    def __init__(self, h5name, dev=None, labels="labels", images="images",
                 start=None, stop=None, label_sel=None, use_geom=False, transform=None,
                 half_precision=False, use_sgnums=False, convert_to_float=False):
        """

        :param h5name: hdf5 master file written by resonet/scripts/merge_h5s.py
        :param dev: pytorch device
        :param labels: path to labels dataset
        :param images: path to images dataset
        :param start: dataset index to begin (default=0)
        :param stop: dataset index to stop (default= all images)
        :param label_names: optional list of labels to select. This requires that the dataset
            specified by labels has names. this can alternatively be a list of numbers
            specifying the indices of the labels dataset
        :param use_geom: if the `geom` dataset exists, then use each iter should return 3-tuple (labels, images, geom)
            Otherwise, each iter returns 2-tuple (images,labels)
            The geom tensor can be used as a secondary input to certain models
        :param use_sgnums:
        :param convert_to_float: automatically convert to float32 if h5 data are in compressed format
        """
        if label_sel is None:
            label_sel = [0]
        elif all([isinstance(l, str) for l in label_sel]):
            label_sel = self._get_label_sel_from_label_names(h5name, labels, label_sel)
        else:
            if not all([isinstance(l, int) for l in label_sel]):
                raise TypeError("label_sel should be all int or all str")
        self.nlab = len(label_sel)
        self.label_sel = label_sel
        self.labels_name = labels
        self.images_name = images
        self.h5name = h5name
        self.use_sgnums = use_sgnums
        self.transform = transform
        self.half_precision = half_precision
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
        self.ops_from_pdb = None
        self.pdb_id_to_num = None
        self.sgnums = None
        self._setup_sgmaps()
        self.convert_to_float = convert_to_float

    def _setup_sgmaps(self):
        if not self.use_sgnums:
            return
        else:
            from resonet.sims import paths_and_const
            assert paths_and_const.SGOP_FILE is not None
            assert os.path.exists(paths_and_const.SGOP_FILE)
        self.ops_from_pdb = np.load(paths_and_const.SGOP_FILE, allow_pickle=True)[()]
        self.pdb_id_to_num = {k: i for i, k in enumerate(self.ops_from_pdb.keys())}

    @staticmethod
    def _get_label_sel_from_label_names(fname, dset_name, label_names):
        label_sel = []
        with h5py.File(fname, "r") as h:
            labels = h[dset_name]
            if "names" not in labels.attrs:
                raise KeyError("the dataset %s in file %s has no `names` attribute" % (dset_name, fname))
            names = list( labels.attrs["names"])
            for name in label_names:
                if name not in names:
                    raise ValueError("label name '%s' is not in 'names' attrs of  dset '%s' (in file %s)" % (name, dset_name, fname))
                idx = names.index(name)
                label_sel.append(idx)
        #TODO what about label_sel ordering?
        return label_sel

    def open(self):
        self.h5 = h5py.File(self.h5name, "r")
        self.images = self.h5[self.images_name]
        assert self.images.dtype in [np.uint16, np.float16, np.float32]
        if self.images.dtype!=np.float32 and not self.convert_to_float:
            raise ValueError("Images should be type float32!")
        self.labels = self.h5[self.labels_name][:, self.label_sel]
        lab_dt = self.labels.dtype
        if not self.half_precision and lab_dt != np.float32:
            self.labels = self.labels.astype(np.float32)
        elif self.half_precision and lab_dt != np.float16:
            self.labels = self.labels.astype(np.float16)
        if self.use_geom:
            geom_dset = self.h5["geom"]
            self.geom = self.get_geom(geom_dset)
            geom_dt = self.geom.dtype
            if not self.half_precision and geom_dt!=np.float32:
                self.geom = self.geom.astype(np.float32)
            elif self.half_precision and geom_dt != np.float16:
                self.geom = self.geom.astype(np.float16)

        if self.use_sgnums:
            self.get_sgnums()

    def get_sgnums(self):
        pdbmap = {i: os.path.basename(f) for i,f in
                   enumerate(self.h5[self.labels_name].attrs['pdbmap'])}
        pdb_i = list(self.h5[self.labels_name].attrs['names']).index('pdb')
        pdb_id_per_img = [pdbmap[i] for i in self.h5['labels'][:, pdb_i].astype(int)]
        self.sgnums = [self.pdb_id_to_num[p] for p in pdb_id_per_img]

    def get_geom(self, geom_dset):
        ngeom = geom_dset.shape[-1]
        inds = list(range(ngeom))
        if "names" in geom_dset.attrs:
            names = list(geom_dset.attrs["names"])

            try:
                inds = [names.index("detdist"),
                        names.index('pixsize'),
                    names.index('wavelen'),
                    names.index("xdim"),
                    names.index("ydim")]
            except ValueError:
                pass
        geom = geom_dset[()][:, inds]
        return geom

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
        img_dat, img_lab = self.images[i + self.start], self.labels[i + self.start]
        if len(img_dat.shape) == 2:
            img_dat = img_dat[None]
        if self.half_precision and not self.images.dtype==np.float16:
            #print("Warning, converting images from float32 to float16. This could slow things down.")
            img_dat = img_dat.astype(np.float16)
        if self.convert_to_float and not self.images.dtype==np.float32:
            img_dat = img_dat.astype(np.float32)
        img_dat = torch.tensor(img_dat).to(self.dev)
        # if we are applying image augmentation
        if self.transform:
            img_dat = self.transform(img_dat)
        img_lab = torch.tensor(img_lab).to(self.dev)
        ret_val = img_dat, img_lab
        if self.use_geom:
            geom_inputs = self.geom[i+self.start]
            geom_inputs = torch.tensor(geom_inputs).to(self.dev)
            ret_val = ret_val + (geom_inputs,)
        if self.use_sgnums:
            sgnums = self.sgnums[i+self.start]
            sgnums = torch.tensor(sgnums).to(self.dev)
            ret_val = ret_val + (sgnums,)
        return ret_val

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

