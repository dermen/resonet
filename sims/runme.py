
from resonet.sims.main import args, run
args = args()

from libtbx.mpi4py import MPI
COMM = MPI.COMM_WORLD

import os
import time


# generate random seeds to each rank
seed_time = args.seed
if args.seed is None:
    if COMM.rank == 0:
        seed_time = int(time.time())
    seed_time = COMM.bcast(seed_time)
seeds = [seed_time + r for r in range(COMM.size)]

# create output directory
if COMM.rank == 0:
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
COMM.barrier()


run(args, seeds, COMM.rank, COMM.size)
