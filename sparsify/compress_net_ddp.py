"""
Distributed Data Parallel training for the sparsify/compress segmentation model.
Uses MPI + NCCL backend, following the td_net.py pattern.

Usage (DDP):
  srun -c2 python compress_net_ddp.py \
    500k_master.h5 --outdir $SCRATCH/compress_train ...

Usage (single-GPU test):
  python compress_net_ddp.py --single_gpu \
    500k_master.h5 --outdir /tmp/compress_test --datafrac 0.001 --nep 2

Usage (with autocorrelation lattice loss):
  python compress_net_ddp.py --single_gpu \
    500k_master.h5 --outdir /tmp/compress_test --datafrac 0.01 --nep 2 \
    --autocorr_weight 3.0 --panels_per_shot 32
"""

import os
import sys
import time
import logging
import socket
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import h5py
import torch
import torch.distributed as td
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler

from resonet.sparsify import sparsify_models
from resonet.loaders import CompressDset, ShotGroupSampler
from resonet.losses import TVLoss, AutocorrStitchLoss

# MPI is optional — only needed for DDP mode
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
except ImportError:
    COMM = None


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


def get_panel_offsets(h5name, panels_per_shot=32):
    """Read panel offsets from H5 geom dataset.

    Returns (offsets, canvas_h, canvas_w) where offsets is (N, 2) of (ystart, xstart).
    """
    with h5py.File(h5name, 'r') as f:
        if 'geom' not in f:
            raise ValueError(
                f"H5 file {h5name} has no 'geom' dataset. "
                "Autocorrelation loss requires panel offsets. "
                "Regenerate data with for_compress.py or disable --autocorr_weight.")
        geom = f['geom']
        names = list(geom.attrs.get('names', []))
        if 'xstart' not in names or 'ystart' not in names:
            raise ValueError(
                f"geom dataset missing xstart/ystart columns. Available: {names}")
        xi = names.index('xstart')
        yi = names.index('ystart')
        g = geom[:panels_per_shot]
        offsets = np.stack([g[:, yi], g[:, xi]], axis=1).astype(int)

        panel_h, panel_w = f['images'].shape[1], f['images'].shape[2]
        canvas_h = int(offsets[:, 0].max()) + panel_h
        canvas_w = int(offsets[:, 1].max()) + panel_w
    return offsets, canvas_h, canvas_w


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
    ap.add_argument("--downsample", type=int, default=1,
                    help="Spatial downsample factor (1=none, 2=half res). "
                         "Uses MaxPool2d on input, nearest upsample on output.")
    ap.add_argument("--FPRate", type=float, default=0.5,
                    help="TVLoss false positive weight (0.5 = Dice)")
    ap.add_argument("--lr_schedule", type=str, default="plateau",
                    choices=["plateau", "cosine", "none"],
                    help="LR schedule after warmup")
    ap.add_argument("--plateau_factor", type=float, default=0.5,
                    help="ReduceLROnPlateau factor")
    ap.add_argument("--plateau_patience", type=int, default=5,
                    help="ReduceLROnPlateau patience")
    ap.add_argument("--save_freq", type=int, default=1,
                    help="Save checkpoint every N epochs (in addition to best)")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--num_workers", type=int, default=2,
                    help="DataLoader workers per GPU")
    ap.add_argument("--single_gpu", action="store_true",
                    help="Run on single GPU without MPI/DDP (for testing)")
    # Autocorrelation lattice loss
    ap.add_argument("--autocorr_weight", type=float, default=0,
                    help="Weight for autocorrelation lattice loss (0=disabled, "
                         "3.0 recommended). Requires geom dataset in H5.")
    ap.add_argument("--panels_per_shot", type=int, default=32,
                    help="Number of panels per detector shot (32 for Eiger 16M)")
    ap.add_argument("--stitch_downsample", type=int, default=4,
                    help="Downsample stitched detector image before FFT "
                         "(saves memory; 4 = ~1k x 1k for Eiger 16M)")
    return ap.parse_args()


def train(args, use_ddp=True):
    if use_ddp:
        from resonet.utils import ddp as ddp_utils
        from resonet.utils import mpi as mpi_utils
        rank = COMM.rank
        world_size = COMM.size
        LOCAL_COMM = mpi_utils.get_host_comm()
        local_rank = LOCAL_COMM.rank

        # Init DDP (same pattern as td_net.py)
        ddp_utils.slurm_init(COMM, mpi_utils.get_host_comm())
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            os.makedirs(args.outdir, exist_ok=True)
        COMM.barrier()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.makedirs(args.outdir, exist_ok=True)

    logger = get_logger(rank, os.path.join(args.outdir, "train.log") if rank == 0 else None)
    use_autocorr = args.autocorr_weight > 0
    pps = args.panels_per_shot

    if rank == 0:
        # Log full run configuration
        logger.info("=" * 60)
        logger.info("compress_net_ddp.py")
        logger.info("=" * 60)
        logger.info(f"Command: {' '.join(sys.argv)}")
        logger.info(f"Working dir: {os.getcwd()}")
        logger.info(f"Host: {socket.gethostname()}")
        logger.info(f"Python: {sys.executable}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA: {torch.version.cuda}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Per-GPU batch size: {args.bs}")
        logger.info(f"Effective batch size: {args.bs * world_size}")
        if use_autocorr:
            logger.info(f"Autocorrelation loss: weight={args.autocorr_weight}, "
                        f"panels_per_shot={pps}, stitch_ds={args.stitch_downsample}")
        logger.info("-" * 60)
        for k, v in vars(args).items():
            logger.info(f"  {k}: {v}")
        logger.info("-" * 60)

        # Save commandline to output folder
        cmd_file = os.path.join(args.outdir, "commandline.txt")
        with open(cmd_file, "w") as o:
            o.write(f"working dir: {os.getcwd()}\n")
            o.write(f"Command: {' '.join(sys.argv)}\n")
            o.write(f"Host: {socket.gethostname()}\n")
            o.write(f"World size: {world_size}\n")
            o.write(f"Effective batch size: {args.bs * world_size}\n\n")
            for k, v in vars(args).items():
                o.write(f"{k}: {v}\n")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build dataset from single H5 file
    with h5py.File(args.h5name, "r") as f:
        total_panels = f['images'].shape[0]
        img_shape = f['images'].shape[1:]  # (H, W)
    if rank == 0:
        logger.info(f"H5 file: {args.h5name}")
        logger.info(f"Total panels: {total_panels}, panel shape: {img_shape}")
        h, w = img_shape
        D = sparsify_models.PaddedEfficientNet.DIVISOR
        pad_h = (D - h % D) % D
        pad_w = (D - w % D) % D
        if pad_h > 0 or pad_w > 0:
            logger.info(f"PaddedEfficientNet will pad: ({h},{w}) -> ({h+pad_h},{w+pad_w}) "
                        f"(divisible by {D} for EfficientNet encoder)")
        else:
            logger.info(f"Panel dims already divisible by {D}")

    use_n = int(total_panels * args.datafrac)

    # When using autocorrelation loss, ensure data is complete shots
    if use_autocorr:
        use_n = (use_n // pps) * pps
        if use_n == 0:
            raise ValueError(f"datafrac={args.datafrac} yields 0 complete shots "
                             f"(need at least {pps} panels)")
        # Enforce batch size = panels_per_shot
        if args.bs != pps:
            if rank == 0:
                logger.info(f"Autocorr mode: overriding bs={args.bs} -> {pps} "
                            f"(must equal panels_per_shot for shot-grouped batching)")
            args.bs = pps

    full_dataset = CompressDset(args.h5name, maximgs=use_n,
                                return_meta=use_autocorr, panels_per_shot=pps)
    if rank == 0 and use_n < total_panels:
        logger.info(f"Using {use_n}/{total_panels} panels (datafrac={args.datafrac})")

    # Data splitting
    if use_autocorr:
        # Shot-level split: keep all panels from a shot together
        n_shots = use_n // pps
        ntrain_shots = int(n_shots * args.trainfrac)
        ntest_shots = n_shots - ntrain_shots

        g = torch.Generator().manual_seed(args.seed)
        shot_perm = torch.randperm(n_shots, generator=g)
        train_shots = shot_perm[:ntrain_shots].sort().values
        test_shots = shot_perm[ntrain_shots:ntrain_shots + ntest_shots].sort().values

        train_panel_indices = []
        for s in train_shots.tolist():
            train_panel_indices.extend(range(s * pps, (s + 1) * pps))
        test_panel_indices = []
        for s in test_shots.tolist():
            test_panel_indices.extend(range(s * pps, (s + 1) * pps))

        train_ds = Subset(full_dataset, train_panel_indices)
        test_ds = Subset(full_dataset, test_panel_indices)
        ntrain = len(train_panel_indices)
        ntest = len(test_panel_indices)

        if rank == 0:
            logger.info(f"Shot-level split: {ntrain_shots} train shots ({ntrain} panels), "
                        f"{ntest_shots} test shots ({ntest} panels)")
    else:
        ntrain = int(use_n * args.trainfrac)
        ntest = use_n - ntrain
        train_ds, test_ds = random_split(full_dataset, [ntrain, ntest],
                                         generator=torch.Generator().manual_seed(args.seed))
        if rank == 0:
            logger.info(f"Train: {ntrain}, Test: {ntest}")

    # Samplers
    if use_autocorr:
        train_sampler = ShotGroupSampler(
            len(train_ds), panels_per_shot=pps,
            rank=rank, world_size=world_size, shuffle=True, seed=args.seed)
        test_sampler = ShotGroupSampler(
            len(test_ds), panels_per_shot=pps,
            rank=rank, world_size=world_size, shuffle=False, seed=args.seed)
    elif use_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_dl = DataLoader(train_ds, batch_size=args.bs, sampler=train_sampler,
                          shuffle=(train_sampler is None),
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=args.bs, sampler=test_sampler,
                         num_workers=args.num_workers, pin_memory=True)

    # Model (padded variant handles variable panel sizes like 512x1028, 514x1030)
    effnet_num = int(args.model.split("-b")[1])
    model = sparsify_models.PaddedEfficientNet(b=effnet_num)
    if args.downsample > 1:
        model = sparsify_models.DownsampleWrapper(model, factor=args.downsample)
        if rank == 0:
            h, w = img_shape
            logger.info(f"DownsampleWrapper: {args.downsample}x downsample "
                        f"({h},{w}) -> ({h//args.downsample},{w//args.downsample}) "
                        f"for model, output upsampled back to ({h},{w})")
    if use_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.float().to(dev)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # LR scheduler
    main_scheduler = None
    prev_lr = args.lr

    if args.lr_schedule == "plateau":
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.plateau_factor,
            patience=args.plateau_patience)
    elif args.lr_schedule == "cosine":
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.nep - args.warmup_epochs)

    # Loss
    loss_fn = TVLoss(falseP_weight=args.FPRate)

    # Autocorrelation lattice loss (training-only, not used at inference)
    autocorr_loss_fn = None
    if use_autocorr:
        panel_offsets, canvas_h, canvas_w = get_panel_offsets(
            args.h5name, panels_per_shot=pps)
        autocorr_loss_fn = AutocorrStitchLoss(
            panel_offsets, canvas_h, canvas_w,
            panels_per_shot=pps,
            stitch_downsample=args.stitch_downsample).to(dev)
        if rank == 0:
            logger.info(f"AutocorrStitchLoss: canvas={canvas_h}x{canvas_w}, "
                        f"stitch_ds={args.stitch_downsample}, "
                        f"weight={args.autocorr_weight}")

    # Helper to get the inner Sequential model regardless of DDP/Downsample wrapping
    def get_inner_model():
        m = model.module if use_ddp else model
        if isinstance(m, sparsify_models.DownsampleWrapper):
            m = m.model  # DownsampleWrapper -> PaddedEfficientNet
        return m.model  # PaddedEfficientNet -> inner Sequential

    # Resume
    start_epoch = 0
    best_val_loss = np.inf
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=dev, weights_only=True)
        get_inner_model().load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if main_scheduler is not None and 'scheduler_state_dict' in ckpt:
            main_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('loss', np.inf)
        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    for epoch in range(start_epoch, args.nep):
        t_epoch = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Warmup LR
        if epoch < args.warmup_epochs:
            warmup_frac = (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * warmup_frac

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_tv_sum = 0.0
        train_ac_sum = 0.0
        train_batches = 0
        for i_batch, batch_data in enumerate(train_dl):
            if use_autocorr:
                img, lab, shot_idx, panel_idx = batch_data
            else:
                img, lab = batch_data
            img = img.float().to(dev)
            lab = lab.float().to(dev)

            optimizer.zero_grad()
            out = model(img)
            # Squeeze channel dim: (B, 1, H, W) -> (B, H, W) to match lab shape
            out = out.squeeze(1)

            tv_loss = loss_fn(out, lab)
            if use_autocorr:
                panel_idx = panel_idx.to(dev)
                ac_loss = autocorr_loss_fn(out, lab, panel_idx)
                loss = tv_loss + args.autocorr_weight * ac_loss
                train_ac_sum += ac_loss.item()
            else:
                loss = tv_loss

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_tv_sum += tv_loss.item()
            train_batches += 1

            if rank == 0:
                extra = ""
                if use_autocorr:
                    extra = (f" tv={tv_loss.item():.6f}"
                             f" ac={ac_loss.item():.6f}")
                print(f"  Ep {epoch+1} train: {i_batch+1}/{len(train_dl)} "
                      f"batch_loss={loss.item():.6f} "
                      f"epoch_avg={train_loss_sum/train_batches:.6f}"
                      f"{extra}",
                      flush=True)

        # All-reduce train loss (DDP only)
        if use_ddp:
            if use_autocorr:
                train_loss_tensor = torch.tensor(
                    [train_loss_sum, train_tv_sum, train_ac_sum, train_batches],
                    device=dev)
            else:
                train_loss_tensor = torch.tensor(
                    [train_loss_sum, train_batches], device=dev)
            td.all_reduce(train_loss_tensor)
            if use_autocorr:
                nb = train_loss_tensor[3]
                avg_train_loss = (train_loss_tensor[0] / nb).item()
                avg_train_tv = (train_loss_tensor[1] / nb).item()
                avg_train_ac = (train_loss_tensor[2] / nb).item()
            else:
                avg_train_loss = (train_loss_tensor[0] / train_loss_tensor[1]).item()
        else:
            avg_train_loss = train_loss_sum / train_batches if train_batches > 0 else 0.0
            if use_autocorr:
                avg_train_tv = train_tv_sum / train_batches if train_batches > 0 else 0.0
                avg_train_ac = train_ac_sum / train_batches if train_batches > 0 else 0.0

        if rank == 0:
            print()

        # --- Eval ---
        model.eval()
        test_loss_sum = 0.0
        test_tv_sum = 0.0
        test_ac_sum = 0.0
        test_batches = 0
        with torch.no_grad():
            for batch_data in test_dl:
                if use_autocorr:
                    img, lab, shot_idx, panel_idx = batch_data
                else:
                    img, lab = batch_data
                img = img.float().to(dev)
                lab = lab.float().to(dev)

                out = model(img)
                out = out.squeeze(1)
                tv_loss = loss_fn(out, lab)

                if use_autocorr:
                    panel_idx = panel_idx.to(dev)
                    ac_loss = autocorr_loss_fn(out, lab, panel_idx)
                    loss = tv_loss + args.autocorr_weight * ac_loss
                    test_ac_sum += ac_loss.item()
                else:
                    loss = tv_loss

                test_loss_sum += loss.item()
                test_tv_sum += tv_loss.item()
                test_batches += 1

        if use_ddp:
            if use_autocorr:
                test_loss_tensor = torch.tensor(
                    [test_loss_sum, test_tv_sum, test_ac_sum, test_batches],
                    device=dev)
            else:
                test_loss_tensor = torch.tensor(
                    [test_loss_sum, test_batches], device=dev)
            td.all_reduce(test_loss_tensor)
            if use_autocorr:
                nb = test_loss_tensor[3]
                avg_test_loss = (test_loss_tensor[0] / nb).item()
                avg_test_tv = (test_loss_tensor[1] / nb).item()
                avg_test_ac = (test_loss_tensor[2] / nb).item()
            else:
                avg_test_loss = (test_loss_tensor[0] / test_loss_tensor[1]).item()
        else:
            avg_test_loss = test_loss_sum / test_batches if test_batches > 0 else 0.0
            if use_autocorr:
                avg_test_tv = test_tv_sum / test_batches if test_batches > 0 else 0.0
                avg_test_ac = test_ac_sum / test_batches if test_batches > 0 else 0.0

        t_epoch = time.time() - t_epoch
        current_lr = optimizer.param_groups[0]['lr']

        if rank == 0:
            if use_autocorr:
                logger.info(
                    f"Epoch {epoch+1}/{args.nep} | "
                    f"Train: {avg_train_loss:.6f} (tv={avg_train_tv:.6f} ac={avg_train_ac:.6f}) | "
                    f"Test: {avg_test_loss:.6f} (tv={avg_test_tv:.6f} ac={avg_test_ac:.6f}) | "
                    f"LR: {current_lr:.2e} | Time: {t_epoch:.1f}s")
            else:
                logger.info(f"Epoch {epoch+1}/{args.nep} | "
                            f"Train: {avg_train_loss:.6f} | Test: {avg_test_loss:.6f} | "
                            f"LR: {current_lr:.2e} | Time: {t_epoch:.1f}s")

        # LR schedule (after warmup)
        if epoch >= args.warmup_epochs and main_scheduler is not None:
            if args.lr_schedule == "plateau":
                main_scheduler.step(avg_test_loss)
            else:
                main_scheduler.step()
            # Log LR changes (since verbose was removed from ReduceLROnPlateau)
            new_lr = optimizer.param_groups[0]['lr']
            if rank == 0 and new_lr != prev_lr:
                logger.info(f"LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")
            prev_lr = new_lr

        # Checkpointing (rank 0 only)
        if rank == 0:
            checkpoint = {
                'model_state_dict': get_inner_model().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': main_scheduler.state_dict() if main_scheduler is not None else None,
                'epoch': epoch,
                'loss': avg_test_loss,
                'train_loss': avg_train_loss,
                'model_name': args.model,
                'downsample_factor': args.downsample,
                'world_size': world_size,
                'effective_bs': args.bs * world_size,
            }

            if avg_test_loss < best_val_loss:
                logger.info(f"New best: {best_val_loss:.6f} -> {avg_test_loss:.6f}")
                best_val_loss = avg_test_loss
                checkpoint['loss'] = best_val_loss
                torch.save(checkpoint, os.path.join(args.outdir, "best.wts"))

            if (epoch + 1) % args.save_freq == 0:
                torch.save(checkpoint, os.path.join(args.outdir, f"epoch_{epoch+1}.wts"))

    if rank == 0:
        logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")

    if use_ddp:
        td.destroy_process_group()


if __name__ == "__main__":
    # Peek at --single_gpu before full parse (avoids MPI requirement for single-GPU mode)
    if "--single_gpu" in sys.argv or COMM is None:
        args = parse_args()
        train(args, use_ddp=False)
    else:
        args = None
        if COMM.rank == 0:
            args = parse_args()
        args = COMM.bcast(args)
        train(args, use_ddp=True)
