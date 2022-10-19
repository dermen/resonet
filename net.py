import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import glob
import numpy as np
import torch.optim as optim
import re
import pandas


class Net(nn.Module):

    def __init__(self, device_id=0):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.dev = "cuda:%d" % device_id
        self.conv1 = nn.Conv2d(1, 6, 5, device=self.dev)
        self.conv2 = nn.Conv2d(6, 16, 5, device=self.dev)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120, device=self.dev)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84, device=self.dev)
        self.fc3 = nn.Linear(84, 10, device=self.dev)
        self.fc4 = nn.Linear(10, 1, device=self.dev)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #from IPython import embed;embed();exit()
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
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
            return np.array(imgs), pandas.concat(labels).reset_index(drop=True)
        else:
            img = Image.open(self.fnames[i])
            num = self.nums[i] 
            label = self.prop.query("num==%d" % num)
            img = np.reshape(img.getdata(), self.img_sh).astype(np.float32)
            return img[None,:512,:512], label

    def tensorload(self, dev,batchsize=4, start=0,  Nf=None):
        if Nf is None:
            Nf = len(self.fnames)
        else:
            assert 0 <= Nf <= len(self.fnames)

        while start < Nf:
            count = batchsize
            if start + count > Nf:
                count = Nf - start
            imgs, labels = self[start:start+count]
            imgs = torch.tensor(imgs).to(dev)
            labels = torch.tensor(labels.reso.values[:,None].astype(np.float32)).to(dev)
            start += batchsize
            yield imgs, labels



if __name__=="__main__":

    nety = Net()
    imgs = Images()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(nety.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):

        train_tens = imgs.tensorload(nety.dev, start=0, batchsize=32)

        losses = []
        for i, (data, labels) in enumerate(train_tens):

            optimizer.zero_grad()

            outputs = nety(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if i % 10 == 0:  
                print("Ep:%d, bacth:%d, loss: %.3f" % (epoch, i, np.mean(losses)))
                losses = []

