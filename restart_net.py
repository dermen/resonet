from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("checkpoint", type=str, help=".chkpt file from net.py")
parser.add_argument("--outdir", type=str, help="optional new output folder")
parser.add_argument("--maxepochs", type=int, help="optional new max number of epochs to run")
args = parser.parse_args()

from resonet import net
import torch

cp = torch.load(args.checkpoint, map_location=torch.device("cpu"))

train_kwargs = dict(cp["args"])
train_kwargs["cp"] = cp
if args.maxepochs is not None:
    train_kwargs["max_ep"] = args.maxepochs
if args.outdir is not None:
    train_kwargs["outdir"] = args.outdir
net.do_training(**train_kwargs)
