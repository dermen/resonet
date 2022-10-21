from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("ep", type=int)
parser.add_argument("outdir", type=str)
parser.add_argument("--lr", type=float, default=0.0075)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--bs", type=int,default=64)
parser.add_argument("--loss", type=str, choices=["L1", "L2"], default="L2")
parser.add_argument("--saveFreq", type=int, default=10)
parser.add_argument("--arch", type=str, choices=["le", "res18", "res50"], default="le")
args = parser.parse_args()

import glob
import re
import os
import sys
import h5py
import numpy as np
import pandas
from scipy.stats import pearsonr, spearmanr
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18, resnet50


MX=127
MN=0.04
DETDIST=200
WAVELEN=.977794
PIXSIZE=.6
IMG_SH=546,518
Y,X = np.indices(IMG_SH)
centX, centY = -0.22111771, -0.77670382 
Rad = np.sqrt((Y-centY)**2 + (X-centX)**2)
QMAP = np.sin(0.5*np.arctan(Rad*PIXSIZE/DETDIST))*2/WAVELEN


class RESNetBase(nn.Module):

    def _set_blocks(self):
        self.resnet.conv1 = nn.Conv2d(2, 64, 
            kernel_size=7, stride=2, padding=3,bias=False, device=self.dev)
        self.fc1 = nn.Linear(1000,100, device=self.dev)
        self.fc2 = nn.Linear(100,1, device=self.dev)
        
    def forward (self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RESNet18(RESNetBase):
    def __init__(self, device_id=0):
        super().__init__()
        self.dev = "cuda:%d" % device_id
        self.resnet = resnet18().to(self.dev)
        self._set_blocks()

class RESNet50(RESNetBase):
    def __init__(self, device_id=0):
        super().__init__()
        self.dev = "cuda:%d" % device_id
        self.resnet = resnet50().to(self.dev)
        self._set_blocks()


class LeNet(nn.Module):

    def __init__(self, device_id=0):
        super().__init__()
        self.dev = "cuda:%d" % device_id
        self.conv1 = nn.Conv2d(2, 6, 3, device=self.dev)
        self.conv2 = nn.Conv2d(6, 16, 3, device=self.dev)
        self.conv2_bn = nn.BatchNorm2d(16, device=self.dev)
        self.conv3 = nn.Conv2d(16, 32, 3, device=self.dev)
        self.conv3_bn = nn.BatchNorm2d(32, device=self.dev)

        self.fc1 = nn.Linear(32*62*62, 1000, device=self.dev)
        self.fc1_bn = nn.BatchNorm1d(1000, device=self.dev)
        self.fc2 = nn.Linear(1000, 100, device=self.dev)
        self.fc2_bn = nn.BatchNorm1d(100, device=self.dev)
        self.fc3 = nn.Linear(100, 1, device=self.dev)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        
        #x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        #x = F.max_pool2d(F.relu(self.conv3_bn(self.conv3(x))), 2)
        
        x = torch.flatten(x, 1) 
        
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc1_bn(self.fc1(x)))
        
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc2_bn(self.fc2(x)))
        # FIXME: batchnorm breaks some validation below 
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
    # choices
    ARCHES = {"le": LeNet, "res18": RESNet18, "res50": RESNet50}
    LOSSES = {"L1": nn.L1Loss, "L2": nn.MSELoss}

    nety = ARCHES[args.arch]() 
    criterion = LOSSES[args.loss]()
    optimizer = optim.SGD(nety.parameters(), lr=args.lr, momentum=0.9)

    imgs = H5Images("trainimages.h5") # data loader

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    logname = os.path.join(args.outdir, "commandline.txt")
    with open(logname, "w") as o:
        o.write(" ".join(sys.argv) + "\n")

    acc = 0
    mx_acc = 0
    for epoch in range(args.ep):

        # <><><><><><><><
        #    Trainings 
        # <><><><><><><><>
        train_tens = tensorload(imgs, nety.dev, start=2000, batchsize=args.bs)

        losses = []
        for i, (data, labels) in enumerate(train_tens):

            optimizer.zero_grad()

            outputs = nety(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % 10 == 0 and len(losses)> 1:  
                print("Ep:%d, batch:%d, loss:  %.5f (latest acc=%.2f%%, max acc=%.2f%%)" \
                    % (epoch, i, np.mean(losses), acc, mx_acc))
                losses = []
    
        test_tens = tensorload(imgs, nety.dev, batchsize=2, start=1000, Nload=1000)

        # <><><><><><><><
        #   Validation
        # <><><><><><><><>
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
        mx_acc = max(acc, mx_acc)
        print("Accuracy at Ep%d: %.2f%%, Ave/Stdev accurate labels=%.3f +- %.3f Angstrom" \
            % (epoch, acc, np.mean(good_labels), np.std(good_labels)))

        # compute correlation coefficients
        all_lab = 1/np.array(all_lab) # convert to resolutions
        all_pred = 1/np.array(all_pred) # convert to resolutions
        pear = pearsonr(all_lab, all_pred)[0]
        spear = spearmanr(all_lab, all_pred)[0]

        print("predicted-VS-truth: PearsonR=%.3f%%, SpearmanR=%.3f%%" \
            % (pear*100, spear*100))

        # <><><><><><><><
        #  End Validation
        # <><><><><><><><>

        if (epoch+1)%args.saveFreq==0:
            outname = os.path.join(args.outdir, "nety_ep%d.nn"%epoch)
            torch.save(nety.state_dict(), outname)
    
    outname = os.path.join(args.outdir, "nety_epLast.nn"%epoch)
    torch.save(nety.state_dict(), outname)


if __name__=="__main__":
    main()

#   TODO
#   BINARY IMAGE CLASSIFIER -> get in the ballpark
#   Shell Image regressions -> fine tune using resolution shell

