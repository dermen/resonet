import os
import socket
from mpi4py import MPI

COMM = MPI.COMM_WORLD
NGPU = 8

import torch
import torch.distributed as td

try:
    os.environ["RANK"] =  os.environ["SLURM_PROCID"]
    os.environ["LOCAL_RANK"] =os.environ["SLURM_LOCALID"]
    os.environ["WORLD_SIZE"] =os.environ["SLURM_NTASKS"]
    os.environ["MASTER_PORT"] ="29500" # default from torch launcher
    os.environ["MASTER_ADDR"] = socket.gethostname()
except KeyError:
    print("Run in a slrum environment on NERSC, or else adjust the above env vars for your env!")

if COMM.rank==0:
    print("addr: ", os.environ["MASTER_ADDR"])
    print("port: ", os.environ["MASTER_PORT"])
    print("rank: ", os.environ["RANK"])
    print("local rank: ", os.environ["LOCAL_RANK"])
    print("world size: ",os.environ["WORLD_SIZE"])

COMM.barrier()

td.init_process_group(backend='nccl', init_method="env://")

print("MPI rank: %d . Torch rank:%d" % (COMM.rank, td.get_rank()))

gpu_id = COMM.rank % NGPU

sz = 1024**3
device = "cuda:%d" % gpu_id
print("rank=%d, gpu=%d" %(COMM.rank, gpu_id))
src = torch.zeros(sz).fill_(COMM.rank).to(device)

td.reduce(src, dst=0)

if COMM.rank==0:
    N = COMM.size - 1
    assert torch.sum(src) == N*(N+1)*.5*sz
else:
    # note reduce seems to leave memory unchanged
    assert torch.sum(src) == COMM.rank*sz

print("", flush=True)
COMM.barrier()
if COMM.rank==0:
    print("ok!")
