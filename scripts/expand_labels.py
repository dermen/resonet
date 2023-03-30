import numpy as np
import h5py


def expand(fname):
    """

    :param fname: filename written by resonet.sims.runme_cpu.py
    :return:
    """
    with h5py.File(fname, "r+") as h:
        L = h["labels"][()]
        label_names = h["labels"].attrs["names"]
        print("Found labels for %s." % ", ".join(label_names))

        for i,name in enumerate(label_names):
            delete_dset(h, name)
            data = L[:, i:i+1]
            dt = np.float32
            h.create_dataset(name, data=data, dtype=dt)
            if name == "reso":
                delete_dset(h, "one_over_reso")
                h.create_dataset("one_over_reso", data=1/data, dtype=dt)
            if name == "radius":
                delete_dset(h, "one_over_radius")
                h.create_dataset("one_over_radius", data=1/data, dtype=dt)

    print("Done expanding labels for", fname)


def delete_dset(h5_group, dset_name):
    """

    :param h5_group: h5 file handle or group
    :param dset_name: dataset name to delete
    :return:
    """
    if dset_name in list(h5_group.keys()):
        del h5_group[dset_name]


if __name__=="__main__":
    import sys
    fname = sys.argv[1]
    expand(fname)
