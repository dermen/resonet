from mpi4py import MPI
COMM = MPI.COMM_WORLD
import net
import numpy as np


def get_max_pixel():
    imgs = net.Images()
    Nf = len(imgs.fnames)

    all_max = []
    for i in range(Nf):
        if i % COMM.size != COMM.rank:
            continue
        img,_ = imgs[i]

        max_val = img.max()
        all_max.append(max_val)
        #if COMM.rank==0:
        #    print(Nf-i)

    all_max = COMM.gather(max(all_max))
    if COMM.rank==0:
        all_max = max(all_max)
        print("maximum pixel value=%.2f" % all_max)
    all_max = COMM.bcast(all_max)
    return all_max


def get_mean_pixel(mx=1):
    imgs = net.Images()
    Nf = len(imgs.fnames)

    all_mean = []
    for i in range(Nf):
        if i % COMM.size != COMM.rank:
            continue
        img,_ = imgs[i]

        mean_val = (img/mx).mean()
        all_mean.append(mean_val)
        #if COMM.rank==0:
        #    print(Nf-i)

    all_mean = COMM.gather(np.mean(all_mean))
    if COMM.rank==0:
        all_mean = np.mean(all_mean)
        print("mean max-normalized pixel value=%.2f" % all_mean)
    all_mean = COMM.bcast(all_mean)
    return all_mean


if __name__=="__main__":
    mx=get_max_pixel()
    get_mean_pixel(mx)

