# More details coming soon

# CXODB76 processing

For example, download one of the data files:

```
wget https://www.cxidb.org/data/76/ln84/r0095_2000.h5
```

Then, download the annotations and mask

```
git clone https://github.com/nksauter/fv5080.git
```

Then, get some models

```
wget https://smb.slac.stanford.edu/~resonet/overlapping.nn  # archstring=res34
wget https://smb.slac.stanford.edu/~resonet/resolution.nn  #archstring=res50
wget https://smb.slac.stanford.edu/~resonet/reso_retrained.nn  #archstring=res50
```

Then, run the command

```
libtbx.python ai_pred.py r0095_2000.h5 --reso reso_retrained.nn --multi overlapping.nn --out r95.proc.npz --thresh 200  --mask fv5080/annotation/LN84/mask.pickle
```

