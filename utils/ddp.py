import os
import socket
import torch.distributed as td
from contextlib import closing


def find_free_port():
    #taken from https://stackoverflow.com/a/45690594/2077270
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def slurm_init(WORLD_COMM=None, LOCAL_COMM=None):

    try:
        if WORLD_COMM is None:
            os.environ["RANK"] =  os.environ["SLURM_PROCID"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["MASTER_PORT"] ="29500" # default from torch launcher
            os.environ["MASTER_ADDR"] = socket.gethostname()
        else:
            os.environ["RANK"] =  str(WORLD_COMM.rank)
            os.environ["LOCAL_RANK"] = str(LOCAL_COMM.rank)
            os.environ["WORLD_SIZE"] = str(WORLD_COMM.size)
            port = host = None
            if WORLD_COMM.rank==0: 
                port = str(find_free_port())
                host=socket.gethostname()
            os.environ["MASTER_PORT"] = WORLD_COMM.bcast(port)
            os.environ["MASTER_ADDR"] = WORLD_COMM.bcast(host) 

    except Exception:
        print("Run in a slrum environment on NERSC, or else adjust the above env vars for your env!")

    td.init_process_group(backend='nccl', init_method="env://")
