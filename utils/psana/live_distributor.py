"""
psana2 live-mode distributor for resonet.

Reads Jungfrau events, extracts a random quadrant around beam center,
downsamples it by max-pooling to 512x512, and PUSHes the ready-to-infer
tensor + geometry metadata to GPU worker daemons.

psana2 manages MPI internally — each rank only sees its own events.

Launch:
  srun -n <nranks> resonet.psana.live_distributor \
      --exp mfxl1234 --run 42 \
      --hosts gpu-node1 gpu-node2 --nwork-per-host 4 \
      --wavelen 1.24 --detdist 150

ZMQ pattern: PUSH (connects to worker PULL sockets)
Message format: multipart [metadata_json, image_bytes]
  image is a ready-to-infer 512x512 float32 array (quad-extracted,
  downsampled, masked, clamped, sqrt'd, floored).
"""
import os
import json
import time
import argparse

import numpy as np

# psana2 live-mode tuning (set before psana import)
os.environ.setdefault('PS_R_MAX_RETRIES', '30')
os.environ.setdefault('PS_SMD_N_EVENTS', '1000')

QUAD_NAMES = ["A", "B", "C", "D"]
# rotation k-values so all quads are oriented the same way (matches to_tens)
QUAD_ROT = {"A": 2, "B": 3, "C": 1, "D": 0}


def extract_quad_512(img, mask, cent, ds_fact, quad, maxval=65025):
    """
    Extract a quadrant from a full image, downsample to 512x512,
    apply mask/clamp/sqrt/floor — pure numpy, no torch needed.

    Replicates the logic of resonet.utils.eval_model.to_tens but
    entirely in numpy so the psana nodes don't need torch/GPU.

    Parameters
    ----------
    img : 2D float32 array (full detector image)
    mask : 2D bool array, same shape as img (True = valid pixel)
    cent : (fast_scan_px, slow_scan_px) beam center in pixel coords
    ds_fact : int downsample factor (e.g. 4 for Jungfrau → 512x512 quads)
    quad : str, one of 'A', 'B', 'C', 'D'
    maxval : saturation cutoff

    Returns
    -------
    512x512 float32 numpy array, ready for inference
    """
    n = 512 * ds_fact
    x, y = int(round(cent[0])), int(round(cent[1]))

    if quad == "A":
        subimg = img[y - n:y, x - n:x]
        submask = mask[y - n:y, x - n:x]
    elif quad == "B":
        subimg = img[y - n:y, x:x + n]
        submask = mask[y - n:y, x:x + n]
    elif quad == "C":
        subimg = img[y:n + y, x - n:x]
        submask = mask[y:n + y, x - n:x]
    else:  # D
        subimg = img[y:n + y, x:n + x]
        submask = mask[y:n + y, x:n + x]

    # Apply mask
    subimg = subimg * submask

    # Downsample via numpy max-pool
    h, w = subimg.shape
    h_ds = (h // ds_fact) * ds_fact
    w_ds = (w // ds_fact) * ds_fact
    subimg = subimg[:h_ds, :w_ds]
    subimg = subimg.reshape(h_ds // ds_fact, ds_fact,
                            w_ds // ds_fact, ds_fact).max(axis=(1, 3))

    # Rotate to canonical orientation (matches to_tens rot90)
    k = QUAD_ROT[quad]
    if k > 0:
        subimg = np.rot90(subimg, k=k)

    # Clamp, sqrt, floor
    subimg = np.clip(subimg, 0, maxval)
    subimg = np.floor(np.sqrt(subimg))

    return subimg.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="psana2 live-mode distributor for resonet GPU workers. "
                    "Extracts random quads, downsamples to 512x512, PUSHes to workers.")

    # Experiment / data
    parser.add_argument("--exp", type=str, required=True,
                        help="psana experiment string (e.g. mfxl1234)")
    parser.add_argument("--run", type=int, required=True,
                        help="Run number")
    parser.add_argument("--live", action="store_true",
                        help="Enable psana2 live mode (real-time streaming)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Stop after this many events (None = all)")
    parser.add_argument("--xtc-dir", type=str, default=None,
                        help="Override XTC directory (for offline replay)")

    # Detector
    parser.add_argument("--det-name", type=str, default="jungfrau",
                        help="psana2 detector name (default: jungfrau)")

    # Geometry (optional — override per-event extraction)
    parser.add_argument("--wavelen", type=float, default=None,
                        help="Nominal wavelength in Angstrom (if None, extract from event)")
    parser.add_argument("--detdist", type=float, default=None,
                        help="Nominal detector distance in mm (if None, extract from event)")
    parser.add_argument("--pixsize", type=float, default=0.075,
                        help="Pixel size in mm (default 0.075 for Jungfrau)")
    parser.add_argument("--center-mm", nargs=2, type=float,
                        default=[155.5, 163.5],
                        help="Beam center in mm (fast, slow). Default for Jungfrau 4M.")

    # Downsampling
    parser.add_argument("--ds-factor", type=int, default=4,
                        help="Downsample factor for quad extraction. "
                             "Quad region = 512 * ds_factor pixels from beam center. "
                             "4 for Jungfrau → 2048x2048 quad → maxpool → 512x512.")

    # Worker connectivity
    parser.add_argument("--hosts", type=str, nargs="+", required=True,
                        help="Hostnames of GPU worker nodes")
    parser.add_argument("--nwork-per-host", dest="nwork", type=int, default=4,
                        help="Number of worker tasks per host (default=4)")
    parser.add_argument("--port-base", type=int, default=5550,
                        help="Starting port for workers (must match worker_daemon)")

    args = parser.parse_args()

    # ── psana2 setup ─────────────────────────────────────────────────────────
    from psana import DataSource

    ds_kwargs = dict(exp=args.exp, run=args.run, live=args.live)
    if args.max_events is not None:
        ds_kwargs["max_events"] = args.max_events
    if args.xtc_dir is not None:
        ds_kwargs["dir"] = args.xtc_dir
    ds = DataSource(**ds_kwargs)

    # psana2 manages MPI internally — each rank only sees its own events
    myrank = int(os.environ.get("SLURM_PROCID", 0))

    # ── ZMQ PUSH to all workers ──────────────────────────────────────────────
    import zmq
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.setsockopt(zmq.SNDHWM, 10)  # backpressure

    for host in args.hosts:
        for local_id in range(args.nwork):
            endpoint = f"tcp://{host}:{args.port_base + local_id}"
            sender.connect(endpoint)
            if myrank == 0:
                print(f"  Connected PUSH -> {endpoint}")

    if myrank == 0:
        total_workers = len(args.hosts) * args.nwork
        print(f"Distributor ready, sending to {total_workers} GPU workers")
        print(f"ds_factor={args.ds_factor}, pixsize={args.pixsize} mm")

    # ── Beam center in pixel coords ──────────────────────────────────────────
    cent_px = [x / args.pixsize for x in args.center_mm]

    # ── RNG for random quad selection ────────────────────────────────────────
    rng = np.random.RandomState(myrank)

    # ── Event loop ───────────────────────────────────────────────────────────
    run = next(ds.runs())
    det = run.Detector(args.det_name)

    # Build mask on first valid event (persists for the run)
    det_mask = None

    count = 0
    t_start = time.time()
    t_proc_total = 0

    for i_evt, evt in enumerate(run.events()):
        # Get calibrated image
        img = det.raw.calib(evt)
        if img is None:
            continue
        if img.dtype != np.float32:
            img = img.astype(np.float32)

        # Build mask once: valid pixels = non-negative, not saturated
        # Combine with detector status mask if available
        if det_mask is None:
            det_mask = img >= 0
            try:
                status_mask = det.raw.mask(evt, calib=True, status=True,
                                           edges=True, central=True)
                if status_mask is not None:
                    det_mask = det_mask & status_mask.astype(bool)
            except Exception:
                pass  # fall back to >= 0 mask
            from scipy.ndimage import binary_dilation
            det_mask = ~binary_dilation(~det_mask, iterations=1)
            if myrank == 0:
                npix = det_mask.size
                nvalid = det_mask.sum()
                print(f"Mask built: {nvalid}/{npix} valid pixels "
                      f"({100 * nvalid / npix:.1f}%)")

        # Wavelength
        if args.wavelen is not None:
            wavelen = args.wavelen
        else:
            try:
                from scipy import constants
                en_convert = 1e10 * constants.c * constants.h / constants.electron_volt
                energy = det.raw.photon_energy(evt)
                wavelen = en_convert / energy
            except Exception:
                print(f"[rank {myrank}] Failed to get wavelength for event {i_evt}, skipping")
                continue

        # Detector distance
        if args.detdist is not None:
            detdist = args.detdist
        else:
            print(f"[rank {myrank}] No --detdist provided, skipping")
            continue

        # Extract random quad, downsample to 512x512
        t0 = time.time()
        quad = rng.choice(QUAD_NAMES)
        quad_img = extract_quad_512(img, det_mask, cent_px, args.ds_factor, quad)
        t_proc_total += time.time() - t0

        # Get event timestamp (psana2 provides nanosecond timestamp)
        try:
            timestamp = evt.timestamp
        except Exception:
            timestamp = 0

        # Build metadata
        meta = {
            "detdist": detdist,
            "pixsize": args.pixsize,
            "wavelen": wavelen,
            "ds_stride": args.ds_factor,
            "quad": quad,
            "run": args.run,
            "event": i_evt,
            "timestamp": timestamp,
            "img_shape": list(quad_img.shape),
        }

        # Send multipart: [metadata_json, quad_512x512_bytes]
        sender.send_multipart([
            json.dumps(meta).encode(),
            quad_img.tobytes(),
        ])

        count += 1
        if count % 200 == 0:
            elapsed = time.time() - t_start
            rate = count / elapsed
            print(f"[rank {myrank}] sent {count} events ({rate:.1f} ev/sec, "
                  f"avg proc {1000 * t_proc_total / count:.2f} ms)")

    # ── Summary ──────────────────────────────────────────────────────────────
    t_total = time.time() - t_start
    print(f"[rank {myrank}] Done: {count} events in {t_total:.1f} sec "
          f"({count / max(t_total, 1e-9):.1f} ev/sec)")

    sender.close()
    context.term()


if __name__ == "__main__":
    main()
