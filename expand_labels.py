import numpy as np
import h5py


def expand(fname):
    """

    :param fname: filename written by resonet.sims.runme_cpu.py
    :return:
    """
    with h5py.File(fname, "r+") as h:
        L = h['labels'][()]

        res = L[:, 0:1]
        rad = L[:, 1:2]
        multi = L[:, 2:3]
        for name in ["res", "rad", "multi", "one_over_res"]:
            if name in list(h.keys()):
                del h[name]
        h.create_dataset("res", data=res, dtype=np.float32)
        h.create_dataset("rad", data=rad, dtype=np.float32)
        h.create_dataset("multi", data=multi, dtype=np.float32)
        h.create_dataset("one_over_res", data=1 / res, dtype=np.float32)
    print(fname)


if __name__=="__main__":
    import sys
    fname = sys.argv[1]
    expand(fname)
