
from joblib import Parallel, delayed
import os
import time


def main():
    from resonet.sims.main import args, run
    args = args(use_joblib=True)

    seed_time = args.seed
    if args.seed is None:
        seed_time = int(time.time())

    seeds = [seed_time + j for j in range(args.njobs)]

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    Parallel(n_jobs=args.njobs)(delayed(run)(args, seeds, j, args.njobs) for j in range(args.njobs))


if __name__=="__main__":
    main()
