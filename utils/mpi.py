import socket

from mpi4py import MPI
COMM = MPI.COMM_WORLD

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

