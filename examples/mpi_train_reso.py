
from mpi4py import MPI
COMM = MPI.COMM_WORLD
import numpy as np
import os
import sys
from resonet.net import do_training

NGPU = 4
dev_id = COMM.rank % NGPU
outdir = sys.argv[1]
if COMM.rank==0:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
COMM.barrier()
subdir = os.path.join(outdir, "rank%d" % COMM.rank)

bs = 32
train_start_stop = 2000, 20000
test_start_stop = 0, 2000

#fnames_drops = [(f, d) for f in fnames for d in [True, False]]

#for i_f, (f,d) in enumerate(fnames_drops):
#    if i_f % COMM.size != COMM.rank:
#        continue
fname='/pscratch/sd/d/dermen/pil.close/master.h5'
lr = np.random.uniform(1e-2, 5e-4)
momentum = np.random.uniform(0, 0.99999)
arch = np.random.choice(["res18", "res34"])
label_sel = ["r%d"%x for x in range(1,10)]
eval_only = np.random.choice([True, False])
error = 1

do_training(fname, "labels", "images", subdir,
            max_ep=1000,
            lr=lr, bs=bs, momentum=momentum,
            dev="cuda:%d" % dev_id,
            arch=arch,
            train_start_stop=train_start_stop,
            label_sel=label_sel,
            test_start_stop=test_start_stop,
            use_geom=False,
            dropout=False,
            save_freq=1,
            display=False,
            loglevel="info",
            ori_mode=True,
            error=error,
            eval_mode_only=eval_only)
