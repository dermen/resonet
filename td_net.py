
from mpi4py import MPI
COMM = MPI.COMM_WORLD

from resonet import net
from resonet.utils import ddp, mpi

args = None
if COMM.rank==0:
    args = net.get_parser().parse_args()
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
            lr=args.lr, bs=args.bs, max_ep=args.ep,
            arch=args.arch, loss=args.loss,
            logfile=args.logfile, loglevel=args.loglevel,
            label_sel=args.labelSel, half_precision=args.half,
            display=not args.noDisplay, save_freq=args.saveFreq,
            COMM=COMM, ngpu_per_node=ngpu_per_node,
            use_geom=args.useGeom, weights=args.weights, error=args.error,
            use_transform=args.transform, eval_mode_only=not args.noEvalOnly,
            debug_mode=args.debugMode, ori_mode=args.oriMode, use_sgnums=args.useSGNums,
            manual_seed=args.manualSeed, kernel_size=args.kernelSize, num_fc=args.numFC,
            test_master=args.testMaster)
