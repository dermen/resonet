
from mpi4py import MPI
COMM = MPI.COMM_WORLD

import glob
import os
#fnames = glob.glob(os.environ["CSCRATCH"] + "/resonet.train/reso/MORE_DATA/*A/master.h5")
fnames = glob.glob(os.environ["CSCRATCH"] + "/resonet.train/reso/MORE_DATA/*A/master.hotpix.h5")
#fnames = [f for f in fnames if "1.25A" in f]
#fnames = [f for f in fnames if "1.87A" in f or "3.65A" in f]
print(fnames)


from resonet.net import do_training

NGPU = 8
dev_id = COMM.rank % NGPU
subdir = "hotpix.trial.1"

#import torch
#available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
assert len(fnames) < NGPU #len(available_gpus)
lr = 1e-4
bs = 16
arch = "res50"
loss = "L1"

for i_f, f in enumerate(fnames):
    if i_f % COMM.size != COMM.rank:
        continue

    dirname = os.path.dirname(f)
    outdir = os.path.join(dirname, subdir)

    #do_training(f, "rad", "images", outdir,
    do_training(f, "rad", "images_with_hotpix", outdir,
                lr=lr, bs=bs, ep=100, momentum=0.9,
                dev="cuda:%d" % dev_id,
                arch=arch, loss=loss,
                loglevel="critical" if COMM.rank >0 else "info",
                title="%s (rank %d)" % (os.path.basename(dirname), COMM.rank))
