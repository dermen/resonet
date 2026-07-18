"""
Persistent GPU worker daemon for resonet inference.

Receives ready-to-infer 512x512 quad images + geometry metadata
over ZMQ PULL socket. No psana dependency — all event reading,
quad extraction, and downsampling is done by the live_distributor.

Launch with srun (one task per GPU):
  srun --gpus-per-node=4 --ntasks-per-node=4 resonet.psana.worker_daemon --ndev 4

ZMQ pattern: PULL (receives from live_distributor PUSH sockets)
Message format: multipart [metadata_json, image_bytes]
  metadata = {"detdist": float, "pixsize": float, "wavelen": float,
              "ds_stride": int, "quad": str,
              "run": int, "event": int,
              "img_shape": [int, int]}
  image = 512x512 float32, already quad-extracted, downsampled,
          masked, clamped, sqrt'd, floored — ready for model input.
"""
import os
import argparse
import json

import zmq
import torch
import numpy as np
import socket as pysocket

RESO_MOD = "/sdf/group/lcls/ds/tools/braggsim/reso_mods/resolution.nn"
HOST = pysocket.gethostname()

from resonet.utils.predict_fabio import ImagePredictFabio


def load_resonet_predictor(device, reso_model):
    print(f"[{HOST}-{device}] Loading Resonet model weights...")
    P = ImagePredictFabio(
        reso_model=reso_model,
        multi_model=None,
        ice_model=None,
        counts_model=None,
        reso_arch="res50",
        multi_arch=None,
        ice_arch=None,
        counts_arch=None,
        dev=device,
        use_modern_reso=True,
        B_to_d=None)
    P.gain = 1
    print(f"[{HOST}-{device}] Model loaded and ready!")
    return P


def run_worker(args):
    device_per_node = args.ndev
    global_id = int(os.environ.get("SLURM_PROCID", 0))
    local_id = int(os.environ.get("SLURM_LOCALID", 0))
    gpu_id = global_id % device_per_node
    device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    unique_port = args.port_base + local_id

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    # Load model once into GPU memory
    P = load_resonet_predictor(device, args.reso_model)

    # ZMQ PULL socket — receives work from distributor PUSH sockets
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(f"tcp://*:{unique_port}")

    # Optional PUSH socket for streaming results downstream
    result_sender = None
    if args.result_host:
        result_sender = context.socket(zmq.PUSH)
        result_sender.connect(f"tcp://{args.result_host}:{args.result_port}")

    print(f"[{HOST} | RANK {local_id} on {device_str}] PULL listening on port {unique_port}")

    # Persistent processing loop
    count = 0
    while True:
        # Receive multipart: [metadata_json, quad_512x512_bytes]
        parts = receiver.recv_multipart()
        meta = json.loads(parts[0])
        img_shape = tuple(meta["img_shape"])
        quad_img = np.frombuffer(parts[1], dtype=np.float32).reshape(img_shape)

        detdist = meta["detdist"]
        pixsize = meta["pixsize"]
        wavelen = meta["wavelen"]
        ds_stride = meta["ds_stride"]

        try:
            # Set geometry tensor (detdist, pixsize, wavelen, ds_stride)
            P.geom = torch.tensor(
                [[detdist, pixsize, wavelen, ds_stride]],
                dtype=torch.float32).to(P._dev)

            # Set pixel tensor directly — image is already a ready-to-infer
            # 512x512 quad (extracted, downsampled, masked, sqrt'd, floored
            # by the distributor). Just convert to torch tensor [1, 1, H, W].
            P.pixels = torch.tensor(
                quad_img, dtype=torch.float32
            ).to(P._dev)[None, None]

            resolution = P.detect_resolution()
            status = "OK"
        except Exception as e:
            resolution = -1.0
            status = f"ERROR: {e}"

        count += 1
        if count % 100 == 0:
            print(f"[{HOST}-{device_str}] processed {count} events, "
                  f"last reso={resolution:.2f} Ang")

        # Ship result downstream if configured
        if result_sender is not None:
            result_sender.send_json({
                "node": HOST,
                "gpu": gpu_id,
                "run": meta.get("run", -1),
                "event": meta.get("event", -1),
                "resolution": resolution,
                "status": status,
            })


def main():
    parser = argparse.ArgumentParser(
        description="Persistent GPU worker for resonet inference (psana-free). "
                    "Receives ready-to-infer 512x512 quads over ZMQ PULL.")
    parser.add_argument("--port-base", type=int, default=5550,
                        help="Starting port number. Local rank ID is added.")
    parser.add_argument("--ndev", type=int, default=4,
                        help="GPU devices per node (default=4)")
    parser.add_argument("--reso-model", type=str, default=RESO_MOD,
                        help="Path to resonet resolution model weights")
    parser.add_argument("--result-host", type=str, default=None,
                        help="Host for PUSH result streaming (disabled if None)")
    parser.add_argument("--result-port", type=int, default=5600,
                        help="Port for PUSH result streaming")
    args = parser.parse_args()
    run_worker(args)


if __name__ == "__main__":
    main()
