from mpi4py import MPI
COMM = MPI.COMM_WORLD

import sys
import os
import numpy as np

from resonet.net import do_training

"""
Fit multiple models at once to scan hyper-parameter space. 
Currently supports one training session per GPU
This example is specifically for training a binary classifier, but the loss
function can be modified in order to train other kinds of models.. 
"""

master_f = sys.argv[1]  # master file written by sims/merge_h5s.py
outdir = sys.argv[2]  # path to output folder
NGPU_PER_NODE = 8  # 8 GPUs per node on CORI-GPU


# let root rank define a set of training parameters
lrs = momenta = batch_sizes = arches = None
if COMM.rank==0:
    lr_choices = np.logspace(np.log10(1e-2), np.log10(1e-5), 20)
    lrs = np.random.choice(lr_choices, replace=True, size=COMM.size)
    momenta = np.random.uniform(0.1,1,COMM.size)
    batch_sizes = np.random.choice([8,16,24], replace=True, size=COMM.size)
    arches = np.random.choice(["res50", "res18", "le"], replace=True, size=COMM.size)
# share with other ranks
lrs = COMM.bcast(lrs)
momenta = COMM.bcast(momenta)
arches = COMM.bcast(arches)
batch_sizes = COMM.bcast(batch_sizes)

# create output dir , only do on root rank!
if COMM.rank==0:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # save parameters
    np.savez(os.path.join(outdir, "params"), lrs=lrs, batch_sizes=batch_sizes,
             momenta=momenta, arches=arches)

# the device ID used by this rank
dev_id = COMM.rank % NGPU_PER_NODE

# iterate over the params
params = zip(lrs, momenta, batch_sizes, arches)
for i_param, (lr, momentum, bs, arch) in enumerate(params):
    if i_param % COMM.size != COMM.rank:
        continue

    sub_outdir = os.path.join(outdir, "param%d" % i_param)

    title_str = "%s: lr=%1.3e, m=%.4f, bs=%d" % (arch, lr, momentum, bs)
    do_training(master_f, "multi", "images", sub_outdir,
                lr=lr, bs=int(bs), ep=100, momentum=momentum,
                dev="cuda:%d" % dev_id,
                train_start_stop=(3000,30000),
                test_start_stop=(0,3000),
                arch=arch, loss="BCE2",
                loglevel="info" if COMM.rank==0 else "critical",
                display=False,
                title="%s (p%d)" % (title_str, i_param))
