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
wget https://smb.slac.stanford.edu/~public_html/dermen/models/overlapping.nn
wget https://smb.slac.stanford.edu/~public_html/dermen/models/reso_orig.nn
wget https://smb.slac.stanford.edu/~public_html/dermen/models/reso_retrained.nn
```

Then, run the command

```
libtbx.python ai_pred.py r0095_2000.h5 --reso reso_retrained.nn --multi overlapping.nn --out r95.proc.npz --thresh 200   --mask fv5080/annotation/LN84/mask.pickle
```

