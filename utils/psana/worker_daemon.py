import os
import argparse
import zmq
import torch
import socket
from scipy import constants

RESO_MOD="/sdf/group/lcls/ds/tools/braggsim/reso_mods/resolution.nn"
HOST=socket.gethostname()
BINNING_MAP = {1:8, 2:4, 3:2, 4:1, 5:1, 6:1, 8:1, 10:1}
en_convert = 1e10 * constants.c * constants.h / constants.electron_volt
from resonet.utils.predict_fabio import ImagePredictFabio

def load_resonet_predictor(device):
    print(f"[{device}] Loading Resonet model weights into memory...")
    P = ImagePredictFabio(
        reso_model=RESO_MOD,
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
    
    print(f"[{HOST}-{device}] Model successfully loaded and ready!")

    P.quads = [-1] # -1 means to use a randomized quadrant for each inference
    P.gain = 1
    return P 

def run_worker(args):
    control_host=args.control_host
    control_port=args.control_port
    device_per_node=args.ndev

    global_id = int(os.environ.get("SLURM_PROCID", 0))
    local_id = int(os.environ.get("SLURM_LOCALID", 0))
    gpu_id = global_id % device_per_node
    device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    unique_port = args.port_base + local_id
    
    # Force PyTorch to only see/use the specific GPU assigned to this task rank
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    
    # 2. LOAD MODEL ONCE (Persistent in memory)
    P = load_resonet_predictor(device)
    
    # 3. Setup ZeroMQ
    context = zmq.Context()
    
    # REP socket to receive event tasks from Master
    task_receiver = context.socket(zmq.REP)
    task_receiver.bind(f"tcp://*:{unique_port}")
    
    # PUSH socket to stream resolutions to the Control Room
    #results_sender = context.socket(zmq.PUSH)
    #results_sender.connect(f"tcp://{control_host}:{control_port}")
    
    #hostname = os.environ.get("SLURMD_NODENAME", "unknown_node")
    print(f"[{HOST} | RANK {local_id} on {device_str}] Listening on port {unique_port}...")

    # Load the datasource / run
    #ds = psana.DataSource("exp=%s:run=%d:idx" % (args.expt, args.run))
    #EBeam = psana.Detector("EBeam")
    #run = next(ds.runs())
    #detz_encoder = psana.Detector(detz_addr)
    
    # Persistent Event Processing Loop
    while True:
        # Block until Master sends an event index
        task = task_receiver.recv_json()
        run_num = task["run"]
        event_idx = task["event"]
        
        try:
            # --- RUN RESONET INFERENCE ---

            #if args.nominalWavelen is None:
            #    try:
            #        ev_ebeam = EBeam.get(ev)
            #        energy = ev_ebeam.ebeamPhotonEnergy()
            #        assert energy > 0
            #        wavelen = en_convert/energy
            #    except (AttributeError, KeyError, AssertionError):
            #        print("Failed to extract wavelength from XTC! Provide a nominalWavelen value")
            #        continue
            #else:
            #    wavelen = args.nominalWavelen
            #    
            #JUNG = psana.Detector("jungfrau")
            #img = JUNG.calib(ev)
            #ydim, xdim = img.shape # should be identical for rayonix,  ydim=xdim
            #pixsize = 0.075
            #P.cent = [x/pixsize for x in cent_mm]
            #P.ds_stride = BINNING_MAP[binning]

            #if args.nominalDetz is None:
            #    detz = detz_encoder(ev) + detz_offset
            #else:
            #    detz = args.nominalDetz
            #P.load_image_from_file_or_array(detdist=detz, 
            #    pixsize=pixsize, wavelen=wavelen, raw_image=img)
            #resolution = P.detect_resolution()

            resolution = 2.35  # Mock result
            status = "OK"
        except Exception as e:
            resolution = -1.0
            status = f"ERROR: {str(e)}"
        
        task_receiver.send_json({"status": "READY"})
        
        # Ship result to the Control Room plotter
        #results_sender.send_json({
        #    "node": HOST,
        #    "gpu": gpu_id,
        #    "run": run_num,
        #    "event": event_idx,
        #    "resolution": resolution,
        #    "status": status
        #})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-base", type=int, default=5550, 
                        help="The starting port number. Local rank ID will be added to this.")
    parser.add_argument("--ndev", type=int, default=1, help="GPU devices per node (default=1)")
    # for the plotting tool which we havent made yet!
    parser.add_argument("--control-host", type=str, default="localhost")
    parser.add_argument("--control-port", type=int, default=5555)
    args = parser.parse_args()
    
    
    run_worker(args)


if __name__ == "__main__":
    main()
