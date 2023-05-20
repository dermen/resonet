
from mpi4py import MPI
COMM = MPI.COMM_WORLD

import glob
import os

fnames = '/pscratch/sd/d/dermen/train.3_1.45A_smallMos/master.h5', '/pscratch/sd/d/dermen/train.3_1.45A_allMos_lowBg/master.h5'
from resonet.net import do_training

NGPU = 4
dev_id = COMM.rank % NGPU

subdir = "mpi.trial.2"
assert len(fnames) < NGPU

#python net.py 1000 /pscratch/sd/d/dermen/train.3_1.45A_smallMos/master.h5 /pscratch/sd/d/dermen/train.3_1.45A_smallMos/trial.1 --lr 1e-3 --bs 64 --arch res50 --loss L1 --labelSel one_over_reso --momentum 0.9 --testRange 0 2400 --trainRange 2400 24000 --useGeom --noDisplay --saveFreq 1 

lr = 1e-3
bs = 64
arch = "res50"
loss = "L1"
train_start_stop = 2400, 24000
test_start_stop=0, 2400

fnames_drops = [(f, d) for f in fnames for d in [True, False]]

for i_f, (f,d) in enumerate(fnames_drops):
    if i_f % COMM.size != COMM.rank:
        continue

    dirname = os.path.dirname(f)
    
    outdir = os.path.join(dirname, subdir + ".dp=%s"%d)

    do_training(f, "labels", "images", outdir,
                max_ep=1000,
                lr=lr, bs=bs, momentum=0.9,
                dev="cuda:%d" % dev_id,
                arch=arch, loss=loss,
                train_start_stop=train_start_stop,
                label_sel=["one_over_reso"],
                test_start_stop=test_start_stop,
                use_geom=True, 
                dropout=d,
                save_freq=1,
                display=False,
                loglevel="info")
