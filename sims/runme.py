

def main():
    from resonet.sims.main import args, run
    from libtbx.mpi4py import MPI
    COMM = MPI.COMM_WORLD
    import os
    import time
    args_parsed = args()

    # generate random seeds to each rank
    seed_time = args_parsed.seed
    if args_parsed.seed is None:
        if COMM.rank == 0:
            seed_time = int(time.time())
        seed_time = COMM.bcast(seed_time)
    seeds = [seed_time + r for r in range(COMM.size)]

    # create output directory
    if COMM.rank == 0:
        if not os.path.exists(args_parsed.outdir):
            os.makedirs(args_parsed.outdir)
    COMM.barrier()

    from simtbx.diffBragg.device import DeviceWrapper
    dev_id = COMM.rank % args_parsed.ngpu
    import numpy as np
    gvec = None
    if COMM.rank==0:
        if args_parsed.randAxis:
            gvec = np.random.normal(0,1,3)
    gvec = COMM.bcast(gvec)
    # TODO: remove this, its for debugging!
    #gvec = np.array([-0.11714061589265543, 0.48394403574869455, 0.8672232967186454])
    with DeviceWrapper(dev_id) as _:
        run(args_parsed, seeds, COMM.rank, COMM.size, gvec=gvec)

if __name__=="__main__":
    main()
