import glob
import os
import torch
import dxtbx
from mpi4py import MPI
from resonet.utils import eval_model
from resonet.arches import RESNetAny
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter as arg_formatter
import time
import os
import sys
import h5py
import numpy as np
#import logging

COMM = MPI.COMM_WORLD

def get_parser():
    parser = ArgumentParser(formatter_class=arg_formatter)
    parser.add_argument("model", type=str, help="input model file (.nn)")
    parser.add_argument("indir", type=str, help="input directory for images")
    parser.add_argument("outdir", type=str, help="output directory for predictions")
    parser.add_argument("--netnum",  type=int, default=50, help="ResNet number (18,34,50,101,152)")
    parser.add_argument("--not_inverse", action="store_true", help="whether model does not predict inverse")
    return parser

def load_model(model_name, netnum=50):
    """ adjust the arch, e.g. RESNet50, RESNet18 etc as necessary
    Also, even though we trained on GPU, we predict on CPU so that we can later parallelize prediction across many CPUs (with each process predicting a different image)
    """
    model = RESNetAny(netnum = netnum, dev="cpu")
    state = torch.load(model_name, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model = model.to("cpu")
    model = model.eval()
    return model

def do_prediction(model_dir,indir,outdir,not_inverse = False, netnum = 50):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    model = load_model(model_dir)
    model_predicts_inverse_res = not not_inverse
    all_cbfs = []
    resolution_list = [1.45,1.87,2.5]
    for resolution in resolution_list:
        cbfs = glob.glob(indir+str(resolution)+"A/*cbf")
        all_cbfs += [cbfs]
    fig, axs = plt.subplots(1,3, figsize=(30, 12))
    fig.subplots_adjust(hspace = .5, wspace=.1)
    axs = axs.ravel()
    resolutions = []
    shot_fnames = []
    for i in range(0,len(all_cbfs)):
        all_res = []
        for i_cbf, cbf in enumerate(all_cbfs[i]):
            if i_cbf % COMM.size != COMM.rank:
                continue
            # used for testing
            #if i_cbf == 10:
            #   break
            loader = dxtbx.load(cbf)
            img = loader.get_raw_data().as_numpy_array()
            mask = img > 0
            is_pilatus = img.shape[1]==2463  # determine camera type, either Pilatus or Eiger
            if is_pilatus:
                tens = eval_model.raw_img_to_tens_pil(img, mask)
            else:
                tens = eval_model.raw_img_to_tens(img, mask)
            res = model(tens).item()
            if model_predicts_inverse_res:
                res = 1/res
            basename = os.path.basename(cbf)
            print("Res = %.3f. File %s (%d / %d)" % (res , basename, i_cbf+1, len(all_cbfs[i])))
            all_res += [res]
            resolutions.append(res)
            shot_fnames.append(cbf)
        axs[i].plot(all_res,'o',label = str(resolution_list[i]))
        axs[i].axhline(y=resolution_list[i], color='r', linestyle='-')

    # save image
    outname = os.path.join(outdir, model_dir+".png")
    plt.savefig(outname)

if __name__ == "__main__":
    # For example, run with:
    #python predict_user_data_test.py ~/capstone-SLAC/resonet/sims/pretrained_transform_error_0.05/nety_ep130.nn /scratch/teo/ prediction
    parser = get_parser()
    args = parser.parse_args()
    #For testing
    #model_name = "~/capstone-SLAC/resonet/sims/pretrained_transform_error_0.05/nety_ep130.nn" 
    #indir = "/scratch/teo"
    #outdir = "predictions"
    do_prediction(model_dir = args.model,
                  indir = args.indir,
                  outdir = args.outdir,
                  not_inverse= args.not_inverse, 
                  netnum = args.netnum)
