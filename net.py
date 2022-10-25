from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("ep", type=int)
    parser.add_argument("input", type=str)
    parser.add_argument("outdir", type=str)
    parser.add_argument("--lr", type=float, default=0.000125)
    parser.add_argument("--noDisplay", action="store_true")
    parser.add_argument("--bs", type=int,default=16)
    parser.add_argument("--loss", type=str, choices=["L1", "L2"], default="L1")
    parser.add_argument("--saveFreq", type=int, default=10)
    parser.add_argument("--arch", type=str, choices=["le", "res18", "res50"], 
                        default="res50")
    parser.add_argument("--loglevel", type=str, 
            choices=["debug", "info", "critical"], default="info")
    parser.add_argument("--logfile", type=str, default="train.log")
    parser.add_argument("--quickTest", action="store_true")
    parser.add_argument("--labelName", type=str, default="labels")
    args = parser.parse_args()
    if hasattr(args, "h") or hasattr(args, "help"):
        parser.print_help()
        sys.exit()

    return parser.parse_args()

import glob
from abc import abstractmethod
import re
import os
import sys
import h5py
import numpy as np
import pandas
import logging
from itertools import chain
from scipy.stats import pearsonr, spearmanr
from PIL import Image
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18, resnet50
from matplotlib.ticker import FormatStrFormatter


def get_logger(filename=None, level="info"):
    logger = logging.getLogger("resonet")
    levels = {"info": 20, "debug": 10, "critical": 50}
    logger.setLevel(levels[level])

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    console.setLevel(levels[level])
    logger.addHandler(console)

    if filename is not None:
        logfile = logging.FileHandler(filename)
        logfile.setFormatter(logging.Formatter("%(asctime)s >>  %(message)s"))
        logfile.setLevel(levels[level])
        logger.addHandler(logfile)
    return logger


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
        self.fc2 = nn.Linear(100, self.nout, device=self.dev)
        
    def forward (self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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


class RESNet18(RESNetBase):
    def __init__(self, device_id=0, nout=1):
        super().__init__()
        self.dev = "cuda:%d" % device_id
        self.nout = nout
        self.resnet = resnet18().to(self.dev)
        self._set_blocks()


class RESNet50(RESNetBase):
    def __init__(self, device_id=0, nout=1):
        super().__init__()
        self.dev = "cuda:%d" % device_id
        self.nout = nout
        self.resnet = resnet50().to(self.dev)
        self._set_blocks()


class LeNet(nn.Module):

    def __init__(self, device_id=0, nout=1):
        super().__init__()
        self.dev = "cuda:%d" % device_id
        self.nout = nout
        self.conv1 = nn.Conv2d(2, 6, 3, device=self.dev)
        self.conv2 = nn.Conv2d(6, 16, 3, device=self.dev)
        self.conv2_bn = nn.BatchNorm2d(16, device=self.dev)
        self.conv3 = nn.Conv2d(16, 32, 3, device=self.dev)
        self.conv3_bn = nn.BatchNorm2d(32, device=self.dev)

        self.fc1 = nn.Linear(32*62*62, 1000, device=self.dev)
        self.fc1_bn = nn.BatchNorm1d(1000, device=self.dev)
        self.fc2 = nn.Linear(1000, 100, device=self.dev)
        self.fc2_bn = nn.BatchNorm1d(100, device=self.dev)
        self.fc3 = nn.Linear(100, self.nout, device=self.dev)
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
        self.props = ["reso"]

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
            labels = labels[self.props].values.astype(np.float32)
            if "reso" in self.props:
                i_reso = self.props.index("reso")
                labels[:,i_reso] = 1/labels[:,i_reso]
            return np.array(imgs), labels
        else:
            img = Image.open(self.fnames[i])
            num = self.nums[i] 

            label = self.prop.query("num==%d" % num)
            img = np.reshape(img.getdata(), self.img_sh).astype(np.float32)
            mask = self.load_mask(self.fnames[i])
            imgQ = np.zeros_like(img)
            imgQ[mask] = QMAP[mask]
            combined_imgs = np.array([img, QMAP])[:,:512,:512]
            return combined_imgs, label 
        

    @staticmethod
    def load_mask(f):
        maskdir = os.path.join( os.path.dirname(f), "masks")
        maskname = os.path.join(maskdir, os.path.basename(f).replace(".png", ".npy"))
        mask = np.load(maskname)
        return mask


class H5Images:
    def __init__(self, h5name, labels=None):
        if labels is None:
            labels = "labels"
        self.h5 = h5py.File(h5name, "r")
        self.images = self.h5["images"]
        self.labels = self.h5[labels]

    @property
    def nlab(self):
        return self.labels.shape[-1]

    @property
    def total(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        if isinstance(i, slice):
            data, lab= self.images[i], self.labels[i]
        elif isinstance(i, int):
            data, lab= self.images[i:i+1], self.labels[i:i+1]
        else:
            data, lab= self.images[i], self.labels[i]
        return data, lab

def tensorload(images, dev, batchsize=4, start=0,  Nload=None, norm=False, seed=None):
    """images is a specialized class with a `total` property, and
    a specialized getitem method (e.g. Images defined above)
    """
    np.random.seed(seed)
    if Nload is None:
        Nload = images.total - start

    stop = start + Nload
    assert start < stop <= images.total

    inds = np.arange(start, stop)
    nroll = np.random.randint(0, len(inds))
    nbatch = int(len(inds)/batchsize)
    batches = np.array_split( np.roll(inds, nroll), nbatch)
    batch_order = np.random.permutation(len(batches))
    for i_batch in batch_order:
        batch = batches[i_batch]
            
        if start in set(batch):
            istart = np.where(batch==start)[0][0]
            if istart==0:
                imgs, labels = images[batch[0]:batch[-1]] 
            else:
                slc_left = slice(batch[0], batch[istart-1]+1, 1)
                slc_right = slice(batch[istart], batch[-1], 1)
                imgs_left, labels_left = images[slc_left]
                imgs_right, labels_right = images[slc_right]
                
                imgs = np.append(imgs_left, imgs_right, axis=0)
                labels = np.append(labels_left, labels_right, axis=0)
        else:
            imgs, labels = images[batch[0]:batch[-1]] 

        if norm:
            imgs /= MX
            imgs -= MN
        imgs = torch.tensor(imgs[:,:,:512,:512]).to(dev)
        labels = torch.tensor(labels).to(dev)
        yield imgs, labels



def validate(input_tens, model, epoch, criterion):
    """
    tens is return value of tensorloader
    TODO make validation multi-channel (e.g. average accuracy over all labels)
    """
    logger = logging.getLogger("resonet")

    total = 0
    nacc = 0 # number of accurate predictions
    all_lab = []
    all_pred = []
    all_loss = []
    for i,(data,labels) in enumerate(input_tens):
        print("validation batch %d"% i,end="\r", flush=True)
        pred = model(data)
        
        all_lab += [[l.item() for l in labs] for labs in labels]
        all_pred += [[p.item() for p in preds] for preds in pred]

        loss = criterion(labels, pred)
        all_loss.append(loss.item())

        errors = (pred-labels).abs()/labels
        is_accurate = errors < 0.1

        nacc += is_accurate.all(dim=1).sum().item()

        total += len(labels)
        
    acc = nacc / total*100.
    all_lab = np.array(all_lab).T
    all_pred = np.array(all_pred).T
    pears = [pearsonr(L,P)[0] for L,P in zip(all_lab, all_pred)]
    spears = [spearmanr(L,P)[0] for L,P in zip(all_lab, all_pred)]
    logger.info("\taccuracy at Ep%d: %.2f%%" \
        % (epoch, acc))
    for pear, spear in zip(pears, spears):
        logger.info("\tpredicted-VS-truth: PearsonR=%.3f%%, SpearmanR=%.3f%%" \
            % (pear*100, spear*100))
    ave_loss = np.mean(all_loss)
    return acc, ave_loss, all_lab, all_pred


def plot_acc(ax, idx, acc, epoch):
    lx, ly = ax.lines[idx].get_data()
    ax.lines[idx].set_data(np.append(lx, epoch),np.append(ly, acc) )
    if epoch==0:
        ax.set_ylim(acc*0.97,acc*1.03)
    else:
        ax.set_ylim(min(min(ly), acc)*0.97, max(max(ly), acc)*1.03)


def save_results_fig(outname, test_lab, test_pred):
    for i_prop in range(test_lab.shape[0]):
        plt.figure()
        plt.plot(test_lab[i_prop], test_pred[i_prop], '.')
        plt.title("Learned property %d"% i_prop)
        plt.xlabel("truth", fontsize=16)
        plt.ylabel("prediction", fontsize=16)
        plt.gca().tick_params(labelsize=12)
        plt.gca().grid(lw=0.5, ls="--")
        plt.subplots_adjust(bottom=.13, left=0.12, right=0.96, top=0.91)
        plt.savefig(outname.replace(".nn", "_results%d.png" % i_prop))
        plt.close()

    with h5py.File(outname.replace(".nn", "_predictions.h5"), "w") as h:
        h.create_dataset("test_pred",data= test_pred)
        h.create_dataset("test_lab", data=test_lab)


def set_ylims(ax):
    ax.set_ylim(min([min(axl.get_data()[1]) for axl in ax.lines])*0.97,
                max([max(axl.get_data()[1]) for axl in ax.lines])*1.03)


def setup_subplots():
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(6.5,5.5))
    ms=8  # markersize
    ax0.tick_params(labelsize=12)
    ax1.tick_params(labelsize=12)
    ax0.grid(1, ls='--')
    ax1.grid(1, ls='--')
    ax0.set_ylabel("loss", fontsize=16)
    ax0.set_xticklabels([])
    ax1.set_xlabel("epoch", fontsize=16)
    ax1.set_ylabel("score (%)", fontsize=16)
    ax1.plot([],[], "tomato", marker="s",ms=ms, label="test")
    ax1.plot([],[], "C0", marker="o", ms=ms,label="train")
    ax0.plot([],[], color='tomato',marker='s', ms=ms,lw=2, label="test")
    ax0.plot([],[], color='C0',marker='o', lw=2, ms=ms,label="train")
    ax0.plot([],[], "C2", marker="*",ms=ms, label="train-full")
    plt.subplots_adjust(top=0.99,right=0.99,left=0.15, hspace=0.04, bottom=0.12)
    return fig, (ax0, ax1)
            

def update_plots(ax0,ax1, epoch):
    ax0.set_xlim(-0.5, epoch+0.5)
    ax1.set_xlim(-0.5, epoch+0.5)
    set_ylims(ax0)
    set_ylims(ax1)
    ax0.legend(prop={"size":12})
    ax1.legend(prop={"size":12})


def main():
    args = get_args()
    
    assert os.path.exists(args.input)
    imgs = H5Images(args.input, args.labelName) # data loader
    
    # model and criterion choices
    ARCHES = {"le": LeNet, "res18": RESNet18, "res50": RESNet50}
    LOSSES = {"L1": nn.L1Loss, "L2": nn.MSELoss}

    #instantiate model
    nety = ARCHES[args.arch](nout=imgs.nlab) 
    criterion = LOSSES[args.loss]()
    optimizer = optim.SGD(nety.parameters(), lr=args.lr, momentum=0.9)


    # setup recordkeeping
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    logname = os.path.join(args.outdir, args.logfile)
    logger = get_logger(logname, args.loglevel)
    logger.info("==== BEGIN RESONET MAIN ====")
    cmdline = " ".join(sys.argv)
    logger.critical(cmdline)

    # optional plots
    fig, (ax0, ax1) = setup_subplots()

    nety.train()
    acc = 0
    mx_acc = 0
    seeds = np.random.choice(9999999, args.ep, replace=False)
    for epoch in range(args.ep):

        # <><><><><><><><
        #    Trainings 
        # <><><><><><><><>
        train_tens = tensorload(imgs, nety.dev, start=2000, 
                            batchsize=args.bs, seed=seeds[epoch], 
                            Nload=100 if args.quickTest else None)

        losses = []
        all_losses = []
        for i, (data, labels) in enumerate(train_tens):

            optimizer.zero_grad()

            outputs = nety(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            l = loss.item()
            losses.append(l)
            all_losses.append(l)
            if i % 10 == 0 and len(losses)> 1:  
                ave_loss = np.mean(losses)
                logger.info("Ep:%d, batch:%d, loss:  %.5f (latest acc=%.2f%%, max acc=%.2f%%)" \
                    % (epoch, i, ave_loss, acc, mx_acc))
                losses = []
                if not args.noDisplay:
                    plt.draw()
                    plt.pause(0.01)
        
        ave_train_loss = np.mean(all_losses)

        nld = 100 if args.quickTest else 1000
        test_tens = tensorload(imgs, nety.dev, batchsize=2, start=1000, Nload=nld)
        train_tens = tensorload(imgs, nety.dev, batchsize=2, start=2000, Nload=nld)
        # <><><><><><><><
        #   Validation
        # <><><><><><><><>
        nety.eval()
        with torch.no_grad():
            logger.info("Computing test accuracy:")
            acc,test_loss, test_lab, test_pred = validate(test_tens, nety, epoch, criterion)
            logger.info("Computing train accuracy:")
            train_acc,train_loss,_,_ = validate(train_tens, nety, epoch, criterion)

            mx_acc = max(acc, mx_acc)
            
            plot_acc(ax0, 0, test_loss, epoch)
            plot_acc(ax0, 1, train_loss, epoch)
            plot_acc(ax0, 2, ave_train_loss, epoch)
            plot_acc(ax1, 0, acc, epoch)
            plot_acc(ax1, 1, train_acc, epoch)

            update_plots(ax0,ax1, epoch)

            if not args.noDisplay:
                plt.draw()
                plt.pause(0.3)

        # <><><><><><><><
        #  End Validation
        # <><><><><><><><>

        # optional save
        if (epoch+1)%args.saveFreq==0:
            outname = os.path.join(args.outdir, "nety_ep%d.nn"%(epoch+1))
            torch.save(nety.state_dict(), outname)
            plt.savefig(outname.replace(".nn", "_train.png"))
            save_results_fig(outname,test_lab, test_pred) 
            
    # final save! 
    outname = os.path.join(args.outdir, "nety_epLast.nn")
    torch.save(nety.state_dict(), outname)
    plt.savefig(outname.replace(".nn", "_train.png"))
    save_results_fig(outname,test_lab, test_pred) 


if __name__=="__main__":
    main()

#   TODO
#   BINARY IMAGE CLASSIFIER -> get in the ballpark
#   Shell Image regressions -> fine tune using resolution shell
#   Spotfinding -> MultiHeadedAttenton models
