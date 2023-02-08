
import numpy as np
from scipy.stats import pearsonr, spearmanr

from torch.utils.data import DataLoader
import torch

from resonet.loaders import PngDset
from resonet.arches import RESNet50
import torch.nn as nn
import torch.optim as optim


pngdir = "/home/rstofer/dials_project/png"
propfile = "/home/rstofer/dials_project/Downloads/num_reso_mos_B_icy1_icy2_cell_SGnum_pdbid_stolid.txt"

# breakdown of images
# first 8k images for training
# validate on first 1k training images
# test on images 8k-9k (we will tune hyper parameters like learning rate to predict these images well)
# secondary test set will be from 9k+ (these test how generalizable our hyper parameter tuning is)
dev = "cuda:0"
train_imgs = PngDset(pngdir=pngdir, propfile=propfile, start=0, stop=8000, dev=dev)
train_imgs_validate = PngDset(pngdir=pngdir, propfile=propfile, start=0, stop=1000, dev=dev)
test_imgs = PngDset(pngdir=pngdir, propfile=propfile, start=8000, stop=9000, dev=dev)

train_tens = DataLoader(train_imgs, batch_size=16, shuffle=True)
train_tens_validate = DataLoader(train_imgs_validate, batch_size=16, shuffle=False)
test_tens = DataLoader(test_imgs, batch_size=16, shuffle=False)


# instantiate model
nety = RESNet50(nout=1, dev=dev)
criterion = nn.L1Loss()
optimizer = optim.SGD(nety.parameters(), lr=1e-3, momentum=0.9)


def validate(input_tens, model, epoch, criterion):
    """
    Used to gauge how well the model is predicting the data
    :param input_tens: pytorch tensor generator
    :param model:  pytorch model (feed it tensors)
    :param epoch: int, the epoch
    :param criterion: pytorch criterion
    :return: 4-tuple of: (accuracy, loss, labels, predictions)
    """
    total = 0
    nacc = 0  # number of accurate predictions
    all_lab = []
    all_pred = []
    all_loss = []
    for i,(data,labels) in enumerate(input_tens):
        print("validation batch %d"% i,end="\r", flush=True)
        pred = model(data)

        loss = criterion(pred, labels)
        all_loss.append(loss.item())

        errors = (pred-labels).abs()/labels

        is_accurate = errors < .05  # arbitrary!
        nacc += is_accurate.all(dim=1).sum().item()
        total += len(labels)

        all_lab += [[l.item() for l in labs] for labs in labels]
        all_pred += [[p.item() for p in preds] for preds in pred]

    all_lab = np.array(all_lab).T
    all_pred = np.array(all_pred).T
    print("\n")

    acc = nacc / total*100.
    pears = [pearsonr(L,P)[0] for L,P in zip(all_lab, all_pred)]
    spears = [spearmanr(L,P)[0] for L,P in zip(all_lab, all_pred)]
    print("\taccuracy at Ep%d: %.2f%%" \
        % (epoch, acc))
    for pear, spear in zip(pears, spears):
        print("\tpredicted-VS-truth: PearsonR=%.3f%%, SpearmanR=%.3f%%" \
            % (pear*100, spear*100))
    ave_loss = np.mean(all_loss)
    return acc, ave_loss, all_lab, all_pred


nety.train()
num_epoch = 10
acc = 0
mx_acc = 0

for epoch in range(num_epoch):

    # <><><><><><><><
    #   Training
    # <><><><><><><><>

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
        if i % 10 == 0 and len(losses) > 1:

            ave_loss = np.mean(losses)

            print("Ep:%d, batch:%d, loss:  %.5f (latest acc=%.2f%%, max acc=%.2f%%)" \
                        % (epoch, i, ave_loss, acc, mx_acc))
            losses = []

    ave_train_loss = np.mean(all_losses)

    # <><><><><><><><
    #   Validation
    # <><><><><><><><>
    nety.eval()
    with torch.no_grad():
        print("Computing test accuracy:")
        test_acc, test_loss, test_lab, test_pred = validate(test_tens, nety, epoch, criterion)
        print("Computing train accuracy:")
        train_acc, train_loss, _, _ = validate(train_tens_validate, nety, epoch, criterion)

        print("Epoch %d results:" % epoch)
        print("Train loss=%.4f, Train Accuracy=%.4f" % (train_acc, train_loss))
        print("Test loss=%.4f, Train Accuracy=%.4f" % (test_acc, test_loss))

        mx_acc = max(test_acc, mx_acc)

# final save!
torch.save(nety.state_dict(), "example.nn")
