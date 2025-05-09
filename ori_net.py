
from argparse import ArgumentParser
import numpy as np

import h5py

ap = ArgumentParser()
ap.add_argument("master", type=str, help="input master file")
ap.add_argument("best", type=str, help="best model file")
args = ap.parse_args()

import pylab as plt

import torch
from resonet.arches import OriQuatModel
from resonet.utils.orientation import QuatLoss
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import SGD
from resonet.loaders import H5SimDataDset
from torch.utils.data import DataLoader
import pylab as plt


ax = plt.gca()
h5images = "images"
h5label = "labels"
use_geom=True
label_sel = ["r%d"%x for x in range(1,10)]
half_precision=False
use_sgnums=True
nep = 5000
lr=3e-4
img_frac = 1 # fraction of images to read
train_frac=.9  # fraction of read images to train on
bs=36
transform = None

common_args = {"dev": "cpu", "labels": h5label, "images": h5images,
               "use_geom": use_geom, "label_sel": label_sel,
               "half_precision": half_precision,
               "use_sgnums": use_sgnums, "convert_to_float": True}

with h5py.File(args.master, "r") as f:
    total_im = int(f['images'].shape[0] * img_frac)
ntrain = int(train_frac*total_im)
ntest = total_im-ntrain
cuda_dev="cuda:0"

all_imgs = H5SimDataDset(args.master,
                         start=0, stop=ntrain + ntest, transform=transform, **common_args)

quatloss = QuatLoss(sgop_table=all_imgs.ops_from_pdb,
                     pdb_id_to_num=all_imgs.pdb_id_to_num,
                     dev=cuda_dev)

img_sh = all_imgs[0][0].shape[-2:]
assert img_sh[0] % 32 == 0
assert img_sh[1] % 32 == 0
model = OriQuatModel(img_sh=img_sh)

train_imgs, test_imgs = random_split( all_imgs, [ntrain, ntest])
train_loader = DataLoader(train_imgs, shuffle=True, batch_size=bs, num_workers=64, pin_memory=True)
test_loader = DataLoader(test_imgs, shuffle=False, batch_size=bs, num_workers=64, pin_memory=True)

gpu={"device": cuda_dev, "non_blocking": True}

model = model.to(**gpu)

optimizer = SGD(model.parameters(), lr=lr, weight_decay=1e-2)

#scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0)

train_dat, test_dat = [],[]
patience = 20
best_test_loss = np.inf
patience_counter = 0
best_model_state = None
for i_ep in range(nep):
    model.train()
    train_epoch_loss = 0
    for i_train, (imgs,labs, geoms, sgnums) in enumerate(train_loader):
        imgs, labs, geoms, sgnums = imgs.to(**gpu), labs.to(**gpu), geoms.to(**gpu), sgnums.to(**gpu)
        geoms = geoms[:,[0,2]]  # we only want distance and wavelength, in that order
        optimizer.zero_grad()
        pred = model(imgs, geoms)
        loss = quatloss(pred, labs, sgnums=sgnums)
        train_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        print("Ep %d: %d/%d Batch Loss=%.8f; Epoch Loss=%.8f" %
              (i_ep+1, i_train+1, len(train_loader), loss.item(), train_epoch_loss/(i_train+1)),
              end="\r",flush=True)
    train_epoch_loss = train_epoch_loss/ len(train_loader)
    print("")
    #scheduler.step()

    model.eval()
    with torch.no_grad():
        test_epoch_loss =0
        for i_test, (imgs, labs, geoms, sgnums) in enumerate(test_loader):
            imgs, labs, geoms, sgnums = imgs.to(**gpu), labs.to(**gpu), geoms.to(**gpu), sgnums.to(**gpu)
            geoms = geoms[:, [0, 2]]  # we only want distance and wavelength, in that order
            outputs = model(imgs, geoms)
            loss = quatloss(outputs, labs, sgnums=sgnums)
            test_epoch_loss += loss.item()
            print(
                "Ep %d: %d/%d Batch Loss=%.8f , Epoch Loss=%.8f" %
                (i_ep+1, i_test + 1, len(test_loader), loss.item(),test_epoch_loss/(i_test+1)),
                end="\r", flush=True)
        test_epoch_loss = test_epoch_loss/ len(test_loader)
        if test_epoch_loss < best_test_loss:
            patience_counter =0
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), args.best)
            best_test_loss = test_epoch_loss
        else:
            patience_counter += 1
    if patience_counter >= patience:
        print("\nEarly stopping...")
        break
    train_dat.append(train_epoch_loss)
    test_dat.append(test_epoch_loss)

    ax.cla()
    ax.plot( np.arange(i_ep+1)+1, train_dat, label="train")
    ax.plot( np.arange(i_ep+1)+1, test_dat, label="test")
    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    ax.legend()
    plt.draw()
    plt.pause(.001)
    print("")
    print("Train Loss=%.8f, Test Loss=%.8f" % (train_epoch_loss, test_epoch_loss))

