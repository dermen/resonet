import h5py

import sys

def main():
    fname = sys.argv[1]
    h = h5py.File(fname, "r")

    print(f"Contents of {fname}")
    for k in h.keys():
        print(k, "dataset shape:", h[k].shape)
        attrs = h[k].attrs.keys()
        for attr in attrs:
            print(attr, "\n")
            print(h[k].attrs[attr])

if __name__=="__main__":
    main()