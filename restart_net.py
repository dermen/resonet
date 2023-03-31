from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("checkpoint", type=str, help=".chkpt file from net.py")
parser.add_argument("--outdir", type=str, help="optional new output folder")
parser.add_argument("--maxepochs", type=int, help="optional new max number of epochs to run")
args = parser.parse_args()

from mpi4py import MPI
COMM = MPI.COMM_WORLD
from resonet.utils import ddp, mpi
from resonet import net
import torch
from resonet.utils.eval_model import strip_names_in_state


cp = torch.load(args.checkpoint, map_location=torch.device("cpu"))
cp["model_state"] = strip_names_in_state(cp["model_state"])
train_kwargs = dict(cp["args"])

if COMM.size > 1:
    LOCAL_COMM = mpi.get_host_comm()
    ngpu_per_node=LOCAL_COMM.size
    if COMM.rank==0:
        print("GPUs per node: %d" % ngpu_per_node, flush=True)
    ddp.slurm_init(COMM, mpi.get_host_comm())
    train_kwargs["COMM"] = COMM
    train_kwargs["ngpu_per_node"] = ngpu_per_node

train_kwargs["cp"] = cp
if args.maxepochs is not None:
    train_kwargs["max_ep"] = args.maxepochs
if args.outdir is not None:
    train_kwargs["outdir"] = args.outdir
net.do_training(**train_kwargs)
