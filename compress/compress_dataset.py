from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("glob", type=str)
ap.add_argument("modelpath", type=str)
ap.add_argument('--cutoff', default=0.9, type=float)
ap.add_argument("--dev", type=str, default="cpu")
args = ap.parse_args()
assert 0 < args.cutoff < 1

import glob
import dxtbx
import fabio
from resonet.utils.multi_panel import split_eiger_16M_to_panels
import torch
from resonet.compress import compress_models

model = compress_models.load_model(args.modelpath)
model = model.float().to(args.dev)

fnames = glob.glob(args.glob)
print("Found %d files!" % len(fnames))
for f in fnames:
    if f.endswith('.cbf'):
        img = fabio.open(f).data
    else:
        loader = dxtbx.load(f)
        # TODO handle multi shot files or multi panel files
        img = loader.get_raw_data().as_numpy_array()
    _,_,_,panels = split_eiger_16M_to_panels(img)
    panel_peaks = []
    for p in panels:
        p = torch.tensor(p[None,None]).float().to(args.dev)
        out = model(p)
        peaks = (out > args.cutoff)[0,0].detach().cpu().numpy()
        panel_peaks.append(peaks)

    from IPython import embed ;embed()



