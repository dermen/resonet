"""
Distributed Data Parallel training for the sparsify/compress segmentation model.
Uses MPI + NCCL backend, following the td_net.py pattern.

Usage:
  srun -c2 conda run -n simtbx python compress_net_ddp.py \
    500k_master.h5 --outdir $SCRATCH/compress_train ...
"""

from mpi4py import MPI
COMM = MPI.COMM_WORLD

import os
import time
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import h5py
import torch
import torch.distributed as td
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from resonet.sparsify import sparsify_models
from resonet.loaders import CompressDset
from resonet.losses import TVLoss
from resonet.utils import ddp as ddp_utils
from resonet.utils import mpi as mpi_utils


def get_logger(rank, filename=None):
    logger = logging.getLogger(f"compress_ddp_rank{rank}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s >> %(message)s")
    if rank == 0:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(message)s"))
        console.setLevel(logging.INFO)
        logger.addHandler(console)
    if filename is not None and rank == 0:
        fh = logging.FileHandler(filename)
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return logger


def parse_args():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("h5name", type=str,
                    help="Path to the HDF5 file (e.g. 500k_master.h5)")
    ap.add_argument("--outdir", type=str, required=True,
                    help="Output directory for checkpoints and logs")
    ap.add_argument("--nep", default=50, type=int, help="Max epochs")
    ap.add_argument("--lr", default=1e-3, type=float, help="Peak learning rate")
    ap.add_argument("--wd", default=1e-4, type=float, help="Weight decay (AdamW)")
    ap.add_argument("--bs", default=16, type=int, help="Per-GPU batch size")
    ap.add_argument("--warmup_epochs", default=5, type=int, help="LR warmup epochs")
    ap.add_argument("--datafrac", default=1.0, type=float, help="Fraction of data to use")
    ap.add_argument("--trainfrac", default=0.9, type=float, help="Train/test split")
    ap.add_argument("--model", type=str, default="eff-b0",
                    choices=["eff-b0", "eff-b1", "eff-b2", "eff-b3", "eff-b4"])
    ap.add_argument("--patience", type=int, default=15,
                    help="Early stopping patience (in epochs)")
    ap.add_argument("--FPRate", type=float, default=0.5,
                    help="TVLoss false positive weight (0.5 = Dice)")
    ap.add_argument("--lr_schedule", type=str, default="plateau",
                    choices=["plateau", "cosine", "none"],
                    help="LR schedule after warmup")
    ap.add_argument("--plateau_factor", type=float, default=0.5,
                    help="ReduceLROnPlateau factor")
    ap.add_argument("--plateau_patience", type=int, default=5,
                    help="ReduceLROnPlateau patience")
    ap.add_argument("--save_freq", type=int, default=5,
                    help="Save checkpoint every N epochs (in addition to best)")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader workers per GPU")
    return ap.parse_args()


def train(args):
    rank = COMM.rank
    world_size = COMM.size
    LOCAL_COMM = mpi_utils.get_host_comm()
    local_rank = LOCAL_COMM.rank

    # Init DDP
    ddp_utils.slurm_init(COMM, LOCAL_COMM)
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
    COMM.barrier()

    logger = get_logger(rank, os.path.join(args.outdir, "train.log") if rank == 0 else None)

    if rank == 0:
        logger.info(f"World size: {world_size}")
        logger.info(f"Args: {args}")
        logger.info(f"Effective batch size: {args.bs * world_size}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build dataset from single H5 file
    with h5py.File(args.h5name, "r") as f:
        total_panels = f['images'].shape[0]
    if rank == 0:
        logger.info(f"H5 file: {args.h5name}, {total_panels} total panels")

    use_n = int(total_panels * args.datafrac)
    full_dataset = CompressDset(args.h5name, maximgs=use_n)
    if rank == 0 and use_n < total_panels:
        logger.info(f"Using {use_n}/{total_panels} panels (datafrac={args.datafrac})")

    ntrain = int(use_n * args.trainfrac)
    ntest = use_n - ntrain
    train_ds, test_ds = random_split(full_dataset, [ntrain, ntest],
                                     generator=torch.Generator().manual_seed(args.seed))

    if rank == 0:
        logger.info(f"Train: {ntrain}, Test: {ntest}")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_dl = DataLoader(train_ds, batch_size=args.bs, sampler=train_sampler,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=args.bs, sampler=test_sampler,
                         num_workers=args.num_workers, pin_memory=True)

    # Model
    effnet_num = int(args.model.split("-b")[1])
    model = sparsify_models.efficientnet(b=effnet_num)
    model = model.float().to(dev)
    model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # LR scheduler
    warmup_scheduler = None
    main_scheduler = None

    if args.lr_schedule == "plateau":
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.plateau_factor,
            patience=args.plateau_patience, verbose=(rank == 0))
    elif args.lr_schedule == "cosine":
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.nep - args.warmup_epochs)

    # Loss
    loss_fn = TVLoss(falseP_weight=args.FPRate)

    # Resume
    start_epoch = 0
    best_val_loss = np.inf
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=dev)
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('loss', np.inf)
        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    patience_counter = 0

    for epoch in range(start_epoch, args.nep):
        t_epoch = time.time()
        train_sampler.set_epoch(epoch)

        # Warmup LR
        if epoch < args.warmup_epochs:
            warmup_frac = (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * warmup_frac

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for i_batch, (img, lab) in enumerate(train_dl):
            img = img.float().to(dev)
            lab = lab.float().to(dev)

            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, lab)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

            if rank == 0 and (i_batch + 1) % 50 == 0:
                print(f"  Ep {epoch+1} train: {i_batch+1}/{len(train_dl)} "
                      f"batch_loss={loss.item():.6f} "
                      f"epoch_avg={train_loss_sum/train_batches:.6f}",
                      end="\r", flush=True)

        # All-reduce train loss
        train_loss_tensor = torch.tensor([train_loss_sum, train_batches], device=dev)
        td.all_reduce(train_loss_tensor)
        avg_train_loss = (train_loss_tensor[0] / train_loss_tensor[1]).item()

        if rank == 0:
            print()

        # --- Eval ---
        model.eval()
        test_loss_sum = 0.0
        test_batches = 0
        with torch.no_grad():
            for img, lab in test_dl:
                img = img.float().to(dev)
                lab = lab.float().to(dev)
                out = model(img)
                loss = loss_fn(out, lab)
                test_loss_sum += loss.item()
                test_batches += 1

        test_loss_tensor = torch.tensor([test_loss_sum, test_batches], device=dev)
        td.all_reduce(test_loss_tensor)
        avg_test_loss = (test_loss_tensor[0] / test_loss_tensor[1]).item()

        t_epoch = time.time() - t_epoch
        current_lr = optimizer.param_groups[0]['lr']

        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.nep} | "
                        f"Train: {avg_train_loss:.6f} | Test: {avg_test_loss:.6f} | "
                        f"LR: {current_lr:.2e} | Time: {t_epoch:.1f}s")

        # LR schedule (after warmup)
        if epoch >= args.warmup_epochs and main_scheduler is not None:
            if args.lr_schedule == "plateau":
                main_scheduler.step(avg_test_loss)
            else:
                main_scheduler.step()

        # Checkpointing (rank 0 only)
        if rank == 0:
            checkpoint = {
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_test_loss,
                'train_loss': avg_train_loss,
                'model_name': args.model,
                'world_size': world_size,
                'effective_bs': args.bs * world_size,
            }

            if avg_test_loss < best_val_loss:
                logger.info(f"New best: {best_val_loss:.6f} -> {avg_test_loss:.6f}")
                patience_counter = 0
                best_val_loss = avg_test_loss
                checkpoint['loss'] = best_val_loss
                torch.save(checkpoint, os.path.join(args.outdir, "best.wts"))
                torch.save(checkpoint, os.path.join(args.outdir, f"best_ep{epoch+1}.wts"))
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience {patience_counter}/{args.patience}")

            if (epoch + 1) % args.save_freq == 0:
                torch.save(checkpoint, os.path.join(args.outdir, f"epoch_{epoch+1}.wts"))

        # Broadcast patience counter so all ranks agree on early stopping
        patience_counter = COMM.bcast(patience_counter if rank == 0 else None)

        if patience_counter >= args.patience:
            if rank == 0:
                logger.info("Early stopping.")
            break

    if rank == 0:
        logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")

    td.destroy_process_group()


if __name__ == "__main__":
    args = None
    if COMM.rank == 0:
        args = parse_args()
    args = COMM.bcast(args)
    train(args)
