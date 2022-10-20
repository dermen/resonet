import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from PIL import Image
import glob
import numpy as np
import torch.optim as optim
import h5py
import re
import pandas
import os


MX=127
MN=0.04
DETDIST=200
WAVELEN=.977794
PIXSIZE=.6
IMG_SH=546,518
Y,X = np.indices(IMG_SH)
centX, centY = -0.22111771, -0.77670382 #np.load("centerA.npy")
Rad = np.sqrt((Y-centY)**2 + (X-centX)**2)
QMAP = np.sin(0.5*np.arctan(Rad*PIXSIZE/DETDIST))*2/WAVELEN


class Net(nn.Module):

    def __init__(self, device_id=0):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.dev = "cuda:%d" % device_id
        self.conv1 = nn.Conv2d(2, 6, 3, device=self.dev, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 3, device=self.dev, padding=2)
        self.conv3 = nn.Conv2d(16, 32, 3, device=self.dev, padding=2)
        #self.conv4 = nn.Conv2d(32, 64, 3, device=self.dev, padding=2)
        #self.conv5 = nn.Conv2d(64, 128, 3, device=self.dev, padding=2)
        #self.conv6 = nn.Conv2d(128, 256, 3, device=self.dev, padding=2)
        # an affine operation: y = Wx + b
        #self.fc1 = nn.Linear(16 * 128 * 128, 3000, device=self.dev)  # 5*5 from image dimension
        self.fc1 = nn.Linear(32*65*65, 4000, device=self.dev)  # 5*5 from image dimension
        self.fc2 = nn.Linear(4000, 1000, device=self.dev)
        self.fc3 = nn.Linear(1000, 1, device=self.dev)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        #x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        #x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        #x = F.max_pool2d(F.relu(self.conv6(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Images:
    def __init__(self, quad="A"):
        dirname="/global/cfs/cdirs/m3992/png/"
        self.fnames = glob.glob(dirname+"*%s.png"%quad)
        assert self.fnames
        self.nums = [self.get_num(f) for f in self.fnames]
        self.img_sh  = 546, 518

        self.prop = pandas.read_csv(
            dirname+"num_reso_mos_B_icy1_icy2_cell_SGnum_pdbid_stolid.txt", 
            delimiter=r"\s+", 
            names=["num", "reso", "mos", "B", "icy1", "icy2", "cell1", \
                    "cell2", "cell3", "SGnum", "pdbid", "stolid"])

    @property
    def total(self):
        return len(self.fnames)

    @staticmethod
    def get_num(f):
        s = re.search("sim_[0-9]{5}", f)
        num = f[s.start(): s.end()].split("sim_")[1]
        return int(num)
     

    def __getitem__(self, i):
        if isinstance(i, slice):
            imgs = []
            labels = []
            step = 1 if i.step is None else i.step
            start = 0 if i.start is None else i.start
            stop = len(self.fnames) if i.stop is None else i.stop
            if stop > len(self.fnames):
                stop = len(self.fnames)
            assert start >=0 and stop >=0, "only supports positive slices"
            for idx in range(start, stop, step):
                img, label = self[idx]
                imgs.append(img)
                labels.append(label)

            labels = pandas.concat(labels).reset_index(drop=True)
            labels = 1/labels.reso.values.astype(np.float32)[:,None]
            return np.array(imgs), labels
        else:
            img = Image.open(self.fnames[i])
            num = self.nums[i] 
            label = self.prop.query("num==%d" % num)
            img = np.reshape(img.getdata(), self.img_sh).astype(np.float32)
            mask = self.load_mask(self.fnames[i])
            imgQ = np.zeros_like(img)
            imgQ[mask] = QMAP[mask]
            combined_imgs = np.array([img, imgQ])[:,:512,:512]
            return combined_imgs, label 

    @staticmethod
    def load_mask(f):
        maskdir = os.path.join( os.path.dirname(f), "masks")
        maskname = os.path.join(maskdir, os.path.basename(f).replace(".png", ".npy"))
        mask = np.load(maskname)
        return mask


class H5Images:
    def __init__(self, h5name):
        self.h5 = h5py.File(h5name, "r")
        self.images = self.h5["images"]
        self.labels = self.h5["labels"]

    @property
    def total(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.images[i], self.labels[i]
        else:
            return self.images[i:i+1], self.labels[i:i+1]



def tensorload(images, dev, batchsize=4, start=0,  Nload=None, norm=False):
    """images is a specialized class with a `total` property, and
    a specialized getitem method (e.g. Images defined above)
    """
    if Nload is None:
        Nload = images.total - start

    stop = start + Nload
    assert start < stop <= images.total

    while start < stop:
        imgs, labels = images[start:start+batchsize]
        if norm:
            imgs /= MX
            imgs -= MN
        imgs = torch.tensor(imgs).to(dev)
        labels = torch.tensor(labels).to(dev)
        start += batchsize
        yield imgs, labels



def main():
    nety = Net()
    #imgs = Images()
    imgs = H5Images("trainimages.h5")
    criterion = nn.MSELoss()
    optimizer = optim.SGD(nety.parameters(), lr=0.001, momentum=0.9)

    acc = 0
    for epoch in range(100):

        train_tens = tensorload(imgs, nety.dev, start=2000, batchsize=64)

        losses = []
        for i, (data, labels) in enumerate(train_tens):

            optimizer.zero_grad()

            outputs = nety(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % 10 == 0 and len(losses)> 1:  
                print("Ep:%d, batch:%d, loss:  %.5f (latest acc=%.2f%%)" \
                    % (epoch, i, np.mean(losses), acc))
                losses = []
    
        test_tens = tensorload(imgs, nety.dev, batchsize=2, start=1000, Nload=1000)
        Ngood = 0
        total = 0
        good_labels = []
        print("Computing accuracy!")
        all_lab = []
        all_pred = []
        for data,labels in test_tens:
            pred = nety(data)
            
            all_lab += [l.item() for l in labels]
            all_pred += [p.item() for p in pred]

            errors = (pred-labels).abs()/labels
            is_accurate = errors < 0.1

            for l in labels[is_accurate]:
                good_labels.append(1/l.item())

            total += len(labels)
            
        acc = len(good_labels) / total*100.
        print("Accuracy at Ep%d: %.2f%%, Ave/Stdev accurate labels=%.3f +- %.3f Angstrom" \
            % (epoch, acc, np.mean(good_labels), np.std(good_labels)))

        # compute correlation coefficients
        all_lab = 1/np.array(all_lab) # convert to resolutions
        all_pred = 1/np.array(all_pred) # convert to resolutions
        pear = pearsonr(all_lab, all_pred)[0]
        spear = spearmanr(all_lab, all_pred)[0]

        print("predicted-VS-truth: PearsonR=%.3f%%, SpearmanR=%.3f%%" \
            % (pear*100, spear*100))


if __name__=="__main__":
    main()

#   BINARY IMAGE CLASSIFIER -> get in the ballpark
#   Shell Image regressions -> fine tune using resolution shell

