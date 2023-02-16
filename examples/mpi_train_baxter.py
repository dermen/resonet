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
lrs = momenta = batch_sizes = arches = weight_decays = dropout = None
if COMM.rank==0:
    lr_choices = np.logspace(np.log10(1e-3), np.log10(1e-4), 20)
    lrs = np.random.choice(lr_choices, replace=True, size=COMM.size)
    momenta = np.random.uniform(0.4,1,COMM.size)
    weight_decays = np.logspace(np.log10(1e-4), np.log10(1e-3), 100)
    batch_sizes = np.random.choice([8], replace=True, size=COMM.size)
    dropout = np.random.randint(0,2,COMM.size)
    #arches = np.random.choice(["res18", "res34", "res101", "res152"], replace=True, size=COMM.size)
    arches = np.random.choice(["res34", "res18"], replace=True, size=COMM.size)
# share with other ranks
weight_decays = COMM.bcast(weight_decays)
dropout = COMM.bcast(dropout)
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
             momenta=momenta, arches=arches, weight_decays=weight_decays, dropout=dropout)

# the device ID used by this rank
dev_id = COMM.rank % NGPU_PER_NODE

# iterate over the params
params = zip(lrs, momenta, batch_sizes, arches, weight_decays, dropout)
for i_param, (lr, momentum, bs, arch, wd, dr) in enumerate(params):
    if i_param % COMM.size != COMM.rank:
        continue

    sub_outdir = os.path.join(outdir, "param%d" % i_param)

    title_str = "%s: lr=%1.3e, m=%.4f, bs=%d" % (arch, lr, momentum, bs)
    do_training(master_f, "multi", "images", sub_outdir,
                lr=lr, bs=int(bs), ep=100, momentum=momentum,
                weight_decay=wd, dropout=dr,
                dev="cuda:%d" % dev_id,
                train_start_stop=(13000,130000),
                test_start_stop=(0,13000),
                arch=arch, loss="BCE2",
                loglevel="info" if COMM.rank==0 else "critical",
                display=False,
                title="%s (p%d)" % (title_str, i_param))
