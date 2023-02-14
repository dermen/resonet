import h5py

import sys

fname = sys.argv[1]
h = h5py.File(fname, "r")

print(f"Contents of {fname}")
for k in h.keys():
    print(k, "dataset shape:", h[k].shape)

