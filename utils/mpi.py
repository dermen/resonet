import socket

from mpi4py import MPI
COMM = MPI.COMM_WORLD
import numpy as np

from resonet.utils import gpu


def get_host_comm():
    """
    get an MPI communicator for all ranks sharing a specific host
    """
    HOST = socket.gethostname()
    unique_hosts = COMM.gather(HOST)
    HOST_MAP = None
    if COMM.rank==0:
        HOST_MAP = {HOST:i for i,HOST in enumerate(set(unique_hosts))}
    HOST_MAP = COMM.bcast(HOST_MAP)
    HOST_COMM = COMM.Split(color=HOST_MAP[HOST])
    return HOST_COMM


def get_gpu_id_mem(ndev):
    """query the memory per GPU and divide ranks accordingly"""
    devs = { i: gpu.get_mem(i)[0] for i in range(ndev)}
    tot_mem = sum(devs.values())
    fracs = {i:v/tot_mem for i,v in devs.items()}

    ranks = list(range(COMM.size))

    positions = []
    start = 0
    for i,f in fracs.items():
        n = int(np.round(f*len(ranks)))
        positions.append(start+n)
        start += n
    ranks_per_dev = np.split(ranks, positions[:-1])
    
    dev_assign = {}
    for gpu_id, gpu_ranks in enumerate(ranks_per_dev):
        for rank in gpu_ranks:
            dev_assign[rank] = gpu_id
    return dev_assign[COMM.rank]

