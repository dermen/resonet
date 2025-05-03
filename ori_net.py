
from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("master", type=str, help="input master file")
args = ap.parse_args()

import torch
from resonet.arches import OriQuatModel
from resonet.utils.orientation import QuatLoss
from torch.utils.data import random_split
from torch.optim import SGD
from resonet.loaders import H5SimDataDset
from torch.utils.data import DataLoader


dev = "cuda:0"
h5images = "images"
h5label = "labels"
use_geom=True
label_sel = ["r%d"%x for x in range(1,10)]
half_precision=False
use_sgnums=True
nep = 10
lr=1e-3
ntrain = 25
ntest = 5
transform = None

common_args = {"dev": dev, "labels": h5label, "images": h5images,
               "use_geom": use_geom, "label_sel": label_sel,
               "half_precision": half_precision,
               "use_sgnums": use_sgnums, "convert_to_float": True}


all_imgs = H5SimDataDset(args.master,
                         start=0, stop=ntrain + ntest, transform=transform, **common_args)

quatloss = QuatLoss(sgop_table=all_imgs.ops_from_pdb,
                     pdb_id_to_num=all_imgs.pdb_id_to_num,
                     dev=all_imgs.dev)

img_sh = all_imgs[0][0].shape[-2:]
assert img_sh[0] % 32 == 0
assert img_sh[1] % 32 == 0
model = OriQuatModel(img_sh=img_sh)

train_imgs, test_imgs = random_split( all_imgs, [ntrain, ntest])
train_loader = DataLoader(train_imgs, shuffle=True, batch_size=2)
test_loader = DataLoader(test_imgs, shuffle=True, batch_size=2)

model = model.to(dev)

optimizer = SGD(model.parameters(), lr=lr)
for i_ep in range(nep):

    model.train()
    for i_train, (imgs,labs, geoms, sgnums) in enumerate(train_loader):
        geoms = geoms[:,[0,2]]  # we only want distance and wavelength, in that order
        optimizer.zero_grad()
        pred = model(imgs, geoms)
        loss = quatloss(pred, labs, sgnums=sgnums)
        loss.backward()
        optimizer.step()
        print(i_train)

    model.eval()
    with torch.no_grad():
        for i_test, (imgs, labs, geoms, sgnums) in enumerate(test_loader):
            geoms = geoms[:, [0, 2]]  # we only want distance and wavelength, in that order
            outputs = model(imgs, geoms)
            loss = quatloss(outputs, labs, sgnums=sgnums)
            print(i_test)

