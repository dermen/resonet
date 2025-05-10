from argparse import ArgumentParser
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from resonet.compress import compress_models
from resonet.loaders import CompressDset # Assuming this is your custom loader
from resonet.losses import TVLoss, diceLoss
from resonet.net import get_logger
import torch
import numpy as np


def args():
    ap = ArgumentParser()
    ap.add_argument("h5name", type=str, help="Path to the HDF5 file containing the dataset.")
    ap.add_argument("out",  type=str, help="Path to save the best model checkpoint.")
    ap.add_argument("--nep", default=100, type=int, help="Number of epochs for training.")
    ap.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    ap.add_argument("--m", default=0, type=float, help="Momentum for SGD optimizer.") # SGD specific
    ap.add_argument("--wd", default=0, type=float, help="Weight decay for the optimizer.")
    ap.add_argument("--bs", default=10, type=int, help="Batch size.")
    ap.add_argument("--datafrac", default=1, type=float, help="Fraction of the dataset to use.")
    ap.add_argument("--trainfrac", default=.9, type=float, help="Fraction of the used dataset for training (rest is for testing).")
    ap.add_argument("--model", type=str, choices=["eff-b0", "fcn50", "eff-b1", "eff-b2", "eff-b3", "eff-b4", "eff-b5", "eff-b6", "eff-b7"] ,
                    help="model strings (see resonet/compress/compress_models.py)")
    ap.add_argument("--patience", type=int, help="early stop count (this many chances to beat minimum loss)", default=7)
    ap.add_argument("--logfile", type=str, help="Log file", default=None)
    ap.add_argument("--FPRate", type=float, default=0.5, help="increase towards 1 to penalize false positives more, decrease towards 0 to penalize false negatives more, 0.5 means DICE loss")
    parsed_args = ap.parse_args()
    return parsed_args


def train(args):
    logger = get_logger(filename=args.logfile)
    assert 0 < args.trainfrac < 1
    assert 0 < args.datafrac <= 1

    import h5py
    with h5py.File(args.h5name, "r") as f:
        nexample = f['images'].shape[0]
        nexample = int(args.datafrac * nexample)
    ds = CompressDset(args.h5name, maximgs=nexample)
    ntrain = int(nexample * args.trainfrac)
    ntest = nexample - ntrain
    
    train_ds, test_ds = random_split(ds, [ntrain, ntest])
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.bs, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=args.bs)

    patience =args.patience
    patience_counter = 0
    best_val_loss = np.inf
    dev = "cuda:0"

    if args.model.startswith("eff"):
        effnet_num = int(args.model.split("-b")[1])
        model = compress_models.efficientnet(b=effnet_num)
    else:
        model= compress_models.FCN50()
    model = model.float().to(dev)

    #optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.m, weight_decay=args.wd)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.wd)

    LOSS = TVLoss(falseP_weight=args.FPRate)

    train_losses, test_losses = [], [] # Changed names for clarity

    for i_ep in range(args.nep):
        model.train()
        train_epoch_loss = 0
        for i_train, (img, lab) in enumerate(train_dl):
            img = img.float().to(dev)
            lab = lab.float().to(dev) # Ensure lab is [0,1]

            optimizer.zero_grad() # Zero gradients BEFORE forward pass
            out = model(img)
            loss = LOSS(out, lab)
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            print(
                f"Ep {i_ep + 1}: {i_train + 1}/{len(train_dl)} Batch Loss={loss.item():.8f}; Epoch Loss={train_epoch_loss / (i_train + 1):.8f}",
                end="\r",
                flush=True,
            )
        train_epoch_loss /= len(train_dl)
        train_losses.append(train_epoch_loss)
        print()

        model.eval()
        test_epoch_loss = 0
        with torch.no_grad(): # Disable gradient calculations for evaluation
            for i_test, (img, lab) in enumerate(test_dl):
                img = img.float().to(dev)
                lab = lab.float().to(dev)
                out = model(img)
                loss = LOSS(out, lab)
                test_epoch_loss += loss.item()
                print(
                    f"Ep {i_ep + 1}: Validating {i_test + 1}/{len(test_dl)} Batch Loss={loss.item():.8f}; Epoch Loss={test_epoch_loss / (i_test + 1):.8f}",
                    end="\r",
                    flush=True,
                )
        test_epoch_loss /= len(test_dl)
        test_losses.append(test_epoch_loss)
        print()

        logger.info(f"Epoch {i_ep + 1}/{args.nep} -> Train Loss: {train_epoch_loss:.8f}, Test Loss: {test_epoch_loss:.8f}")

        if test_epoch_loss < best_val_loss:
            logger.info(f"Validation loss improved from {best_val_loss:.8f} to {test_epoch_loss:.8f}. Saving new best model...")
            patience_counter = 0
            best_val_loss = test_epoch_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': i_ep,
                'loss': best_val_loss,
                'model_name': args.model,
            }
            torch.save(checkpoint, args.out)
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve. Patience {patience_counter}/{patience}.")
        
        if patience_counter >= patience:
            logger.info("Early stopping condition detected.")
            break

    logger.info("Training finished.")

if __name__ == "__main__":
    A = args()
    train(A)

