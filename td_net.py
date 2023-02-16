
from mpi4py import MPI
COMM = MPI.COMM_WORLD

#import torch.distributed as td

from resonet import net
from resonet.utils import ddp, mpi

args = None
if COMM.rank==0:
    args = net.get_args()
args = COMM.bcast(args)



LOCAL_COMM = mpi.get_host_comm()
ngpu_per_node=LOCAL_COMM.size
if COMM.rank==0:
    print("GPUs per node: %d" % ngpu_per_node, flush=True)

ddp.slurm_init(COMM, mpi.get_host_comm())

net.do_training(args.input, args.labelName, args.imgsName, args.outdir,
            train_start_stop=args.trainRange,
            test_start_stop=args.testRange,
            momentum=args.momentum,
            weight_decay=args.weightDecay, 
            nesterov=args.nesterov, damp=args.damp,
            dropout=args.dropout,
            lr=args.lr, bs=args.bs, ep=args.ep,
            arch=args.arch, loss=args.loss,
            logfile=args.logfile, loglevel=args.loglevel,
            display=True, save_freq=args.saveFreq,
            COMM=COMM, ngpu_per_node=ngpu_per_node)
