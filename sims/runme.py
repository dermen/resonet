

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

    run(args_parsed, seeds, COMM.rank, COMM.size)

if __name__=="__main__":
    main()
