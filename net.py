from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter as arg_formatter


def get_args():
    parser = ArgumentParser(formatter_class=arg_formatter)
    parser.add_argument("ep", type=int, help="number of epochs")
    parser.add_argument("input", type=str, help="input training data h5 file")
    parser.add_argument("outdir", type=str, help="store output files here (will create if necessary)")
    parser.add_argument("--lr", type=float, default=0.000125, help="learning rate (important!)")
    parser.add_argument("--noDisplay", action="store_true", help="dont shot plots")
    parser.add_argument("--bs", type=int,default=16, help="batch size")
    parser.add_argument("--loss", type=str, choices=["L1", "L2", "BCE", "BCE2"], default="L1", help="loss function selector")
    parser.add_argument("--saveFreq", type=int, default=10, help="how often to write the model to disk")
    parser.add_argument("--arch", type=str, choices=["le", "res18", "res50" ,"res50bc", "res34", "res101", "res152"],
                        default="res50", help="architecture selector")
    parser.add_argument("--loglevel", type=str, 
            choices=["debug", "info", "critical"], default="info", help="python logger level")
    parser.add_argument("--logfile", type=str, default="train.log", help="logfile, file basename only, like `log.txt`, will be written to outdir")
    parser.add_argument("--quickTest", action="store_true",help="train/test on 100 image")
    parser.add_argument("--labelName", type=str, default="labels", help="path to training labels (in input h5 file)")
    parser.add_argument("--imgsName", type=str, default="images", help="path to training images (in input h5 file)")
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--weightDecay", default=0, type=float)
    parser.add_argument("--trainRange", type=int, nargs=2, default=None)
    parser.add_argument("--testRange", type=int, nargs=2, default=None)
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD optimizer")
    parser.add_argument("--nesterov", action="store_true", help="use nesterov momentum (SGD)")
    parser.add_argument("--damp", type=float, default=0, help="dampening (SGD)")
    args = parser.parse_args()
    if hasattr(args, "h") or hasattr(args, "help"):
        parser.print_help()
        sys.exit()

    return parser.parse_args()


import time
import os
import sys
import h5py
import numpy as np
import logging
from scipy.stats import pearsonr, spearmanr
import pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex


from resonet.params import ARCHES, LOSSES
from resonet.loaders import H5SimDataDset
from resonet import arches


def get_logger(filename=None, level="info", do_nothing=False):
    """

    :param filename: optionally log to a file
    :param level: logging level of the console (info, debug or critical)
     do_nothing: return a logger that doesnt actually log (for non-root processes)
    :return:
    """
    levels = {"info": 20, "debug": 10, "critical": 50}
    if do_nothing:
        logger = logging.getLogger()
        logger.setLevel(levels["critical"])
        return logger
    logger = logging.getLogger("resonet")
    logger.setLevel(levels["info"])

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    console.setLevel(levels[level])
    logger.addHandler(console)

    if filename is not None:
        logfile = logging.FileHandler(filename)
        logfile.setFormatter(logging.Formatter("%(asctime)s >>  %(message)s"))
        logfile.setLevel(levels["info"])
        logger.addHandler(logfile)
    return logger


def validate(input_tens, model, epoch, criterion, COMM=None):
    """
    tens is return value of tensorloader
    TODO make validation multi-channel (e.g. average accuracy over all labels)
    """
    logger = logging.getLogger("resonet")
    using_bce = str(criterion).startswith("BCE")

    total = 0
    nacc = 0  # number of accurate predictions
    all_lab = []
    all_pred = []
    all_loss = []
    for i,(data,labels) in enumerate(input_tens):
        if COMM is None or COMM.rank==0:
            print("validation batch %d"% i,end="\r", flush=True)
        pred = model(data)

        loss = criterion(pred, labels)
        all_loss.append(loss.item())

        if using_bce:
            pred = torch.round(torch.sigmoid(pred))
        else:
            errors = (pred-labels).abs()/labels
            is_accurate = errors < .3
            nacc += is_accurate.all(dim=1).sum().item()
            total += len(labels)

        all_lab += [[l.item() for l in labs] for labs in labels]
        all_pred += [[p.item() for p in preds] for preds in pred]

    if COMM is not None:
        all_lab = COMM.bcast(COMM.reduce(all_lab))
        all_pred = COMM.bcast(COMM.reduce(all_pred))
        all_loss = COMM.bcast(COMM.reduce(all_loss))

    all_lab = np.array(all_lab).T
    all_pred = np.array(all_pred).T
    if COMM is None or COMM.rank==0:
        print("Number of samples:" , all_lab.shape)
        print("\n")

    if not using_bce:
        acc = nacc / total*100.
        pears = [pearsonr(L,P)[0] for L,P in zip(all_lab, all_pred)]
        spears = [spearmanr(L,P)[0] for L,P in zip(all_lab, all_pred)]
        logger.info("\taccuracy at Ep%d: %.2f%%" \
            % (epoch, acc))
        for pear, spear in zip(pears, spears):
            logger.info("\tpredicted-VS-truth: PearsonR=%.3f%%, SpearmanR=%.3f%%" \
                % (pear*100, spear*100))
        ave_loss = np.mean(all_loss)
        return acc, ave_loss, all_lab, all_pred
    else:
        acc = np.sum(all_pred == all_lab) / all_pred.shape[-1] * 100
        ave_loss = np.mean(all_loss)
        jaccard = BinaryJaccardIndex()(torch.tensor(all_pred), torch.tensor(all_lab))
        logger.info("\taccuracy at Ep%d: %.2f%%" \
                    % (epoch, acc))
        logger.info("\tpredicted-VS-truth: Jaccard=%.3f" % jaccard)
        return acc, ave_loss, all_lab, all_pred


def plot_acc(ax, idx, acc, epoch):
    lx, ly = ax.lines[idx].get_data()
    ax.lines[idx].set_data(np.append(lx, epoch),np.append(ly, acc) )
    if epoch==0:
        ax.set_ylim(acc*0.97,acc*1.03)
    else:
        ax.set_ylim(min(min(ly), acc)*0.97, max(max(ly), acc)*1.03)


def save_results_fig(outname, test_lab, test_pred):
    for i_prop in range(test_lab.shape[0]):
        plt.figure()
        plt.plot(test_lab[i_prop], test_pred[i_prop], '.')
        plt.title("Learned property %d"% i_prop)
        plt.xlabel("truth", fontsize=16)
        plt.ylabel("prediction", fontsize=16)
        plt.gca().tick_params(labelsize=12)
        plt.gca().grid(lw=0.5, ls="--")
        plt.subplots_adjust(bottom=.13, left=0.12, right=0.96, top=0.91)
        plt.savefig(outname.replace(".nn", "_results%d.png" % i_prop))
        plt.close()

    with h5py.File(outname.replace(".nn", "_predictions.h5"), "w") as h:
        h.create_dataset("test_pred",data= test_pred)
        h.create_dataset("test_lab", data=test_lab)


def set_ylims(ax):
    ax.set_ylim(min([min(axl.get_data()[1]) for axl in ax.lines])*0.97,
                max([max(axl.get_data()[1]) for axl in ax.lines])*1.03)


def setup_subplots(title=""):
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(6.5,5.5))
    plt.suptitle(title, fontsize=16)
    ms=8  # markersize
    ax0.tick_params(labelsize=12)
    ax1.tick_params(labelsize=12)
    ax0.grid(1, ls='--')
    ax1.grid(1, ls='--')
    ax0.set_ylabel("loss", fontsize=16)
    ax0.set_xticklabels([])
    ax1.set_xlabel("epoch", fontsize=16)
    ax1.set_ylabel("score (%)", fontsize=16)
    ax1.plot([],[], "tomato", marker="s",ms=ms, label="test")
    ax1.plot([],[], "C0", marker="o", ms=ms,label="train")
    ax0.plot([],[], color='tomato',marker='s', ms=ms,lw=2, label="test")
    ax0.plot([],[], color='C0',marker='o', lw=2, ms=ms,label="train")
    ax0.plot([],[], "C2", marker="None",ms=ms, ls="None")
    plt.subplots_adjust(top=0.94,right=0.99,left=0.15, hspace=0.04, bottom=0.12)
    return fig, (ax0, ax1)
            

def update_plots(ax0,ax1, epoch):
    ax0.set_xlim(-0.5, epoch+0.5)
    ax1.set_xlim(-0.5, epoch+0.5)
    set_ylims(ax0)
    set_ylims(ax1)
    ax0.legend(prop={"size":12})
    ax1.legend(prop={"size":12})


def do_training(h5input, h5label, h5imgs, outdir,
         lr=1e-3, bs=16, ep=100, momentum=0.9,
         weight_decay=0, dropout=False,
         nesterov=False, damp=0,
         arch="res50", loss="L1", dev="cuda:0",
         logfile=None, train_start_stop=None, test_start_stop=None,
         loglevel="info",
         display=True, save_freq=10,
         num_workers=1,
         title=None, COMM=None, ngpu_per_node=1):

    # model and criterion choices

    assert loglevel in ["info", "debug", "critical"]

    assert arch in ARCHES
    assert loss in LOSSES

    if logfile is None:
        logfile = "train.log"

    if train_start_stop is None:
        train_start, train_stop = 2000,15000
    else:
        train_start, train_stop = train_start_stop
    if test_start_stop is None:
        test_start, test_stop = 0, 2000
    else:
        test_start, test_stop = test_start_stop

    # make sure train/test sets dont intersect
    train_rng = range(train_start, train_stop)
    test_rng = range(test_start, test_stop)
    assert not set(train_rng).intersection(test_rng)
    ntest = test_stop - test_start

    assert os.path.exists(h5input)
    if COMM is not None:
        # TODO: assert that e.g. slurm_init has been called (distributed.init_process_group)
        gpuid = COMM.rank % ngpu_per_node
        dev = "cuda:%d" % gpuid

    train_imgs = H5SimDataDset(h5input,  dev=dev, labels=h5label, images=h5imgs,
                           start=train_start, stop=train_stop)
    train_imgs_validate = H5SimDataDset(h5input,dev=dev,  labels=h5label, images=h5imgs,
                                    start=train_start, stop=train_start+ntest)
    test_imgs = H5SimDataDset(h5input, dev=dev, labels=h5label, images=h5imgs,
                          start=test_start, stop=test_stop)

    #instantiate model
    nety = ARCHES[arch](nout=train_imgs.nlab, dev=train_imgs.dev, dropout=dropout)
    if COMM is not None:
        nety = torch.nn.SyncBatchNorm.convert_sync_batchnorm(nety)
        nety = nn.parallel.DistributedDataParallel(nety, device_ids=[gpuid], 
            find_unused_parameters= arch in ["le"])


    criterion = LOSSES[loss]()
    optimizer = optim.SGD(nety.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, dampening=damp )
    #optimizer = optim.Adam(nety.parameters(), lr=lr, weight_decay=weight_decay)
    #sched = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)

    # setup recordkeeping
    if COMM is None or COMM.rank==0:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        logname = os.path.join(outdir, logfile)
        logger = get_logger(logname, loglevel)
        logger.info("==== BEGIN RESONET MAIN ====")
        cmdline = " ".join(sys.argv)
        logger.critical(cmdline)
    else:
        logger = get_logger(do_nothing=True)

    
    if COMM is None or COMM.rank==0:
        # optional plots
        if title is None:
            title = os.path.join(os.path.basename(os.path.dirname(h5input)), 
                        os.path.basename(h5input))
        fig, (ax0, ax1) = setup_subplots(title)

    nety.train()
    acc = 0
    mx_acc = 0

    shuffle = True
    train_sampler = train_validate_sampler = test_sampler = None
    if COMM is not None:
        shuffle = None
        train_sampler = DistributedSampler(train_imgs, rank=COMM.rank, num_replicas=COMM.size) 
        train_validate_sampler = DistributedSampler(train_imgs_validate) 
        test_sampler = DistributedSampler(test_imgs) 
         
    train_tens = DataLoader(train_imgs, batch_size=bs, shuffle=shuffle, 
                        sampler=train_sampler)
    train_tens_validate = DataLoader(train_imgs_validate, batch_size=bs, shuffle=shuffle, 
                        sampler=train_validate_sampler)
    test_tens = DataLoader(test_imgs, batch_size=bs, shuffle=shuffle, sampler=test_sampler)

    nbatch = np.ceil((train_stop - train_start) / bs)
    if COMM is not None:
        nbatch = np.ceil((train_stop - train_start) / bs / COMM.size)

    for epoch in range(ep):


        # <><><><><><><><
        #    Trainings 
        # <><><><><><><><>

        t = time.time()
        losses = []
        all_losses = []

        if display and (COMM is None or COMM.rank==0):
            plt.draw()
            plt.pause(0.01)
        
        if COMM is not None:  # or if train_tens.sampler is not None
            train_tens.sampler.set_epoch(epoch)

        for i, (data, labels) in enumerate(train_tens):
            
            if COMM is None or COMM.rank==0:
                print("Training Epoch %d batch %d/%d" \
                    % (epoch+1, i+1, nbatch), flush=True)

            optimizer.zero_grad()

            outputs = nety(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #l = loss.item()
            #losses.append(l)
            #all_losses.append(l)
            #if i % 10 == 0 and len(losses)> 1:
            #
            #    ave_loss = np.mean(losses)

            #    logger.info("Ep:%d, batch:%d, loss:  %.5f (latest acc=%.2f%%, max acc=%.2f%%)" \
            #        % (epoch, i, ave_loss, acc, mx_acc))
            #    losses = []
        
        #ave_train_loss = np.mean(all_losses)
        t = time.time()-t
        if COMM is None or COMM.rank==0:
            print("Traing time: %.4f sec" % t, flush=True)

        # <><><><><><><><
        #   Validation
        # <><><><><><><><>
        nety.eval()
        with torch.no_grad():
            logger.info("Computing test accuracy:")
            acc,test_loss, test_lab, test_pred = validate(test_tens, nety, epoch, criterion, COMM)
            logger.info("Computing train accuracy:")
            train_acc,train_loss,_,_ = validate(train_tens_validate, nety, epoch, criterion, COMM)

            mx_acc = max(acc, mx_acc)
            
            if COMM is None or COMM.rank==0:    
                plot_acc(ax0, 0, test_loss, epoch)
                plot_acc(ax0, 1, train_loss, epoch)
                plot_acc(ax0, 2, train_loss, epoch)
                plot_acc(ax1, 0, acc, epoch)
                plot_acc(ax1, 1, train_acc, epoch)

                update_plots(ax0,ax1, epoch)

                if display:
                    plt.draw()
                    plt.pause(0.3)

        # <><><><><><><><
        #  End Validation
        # <><><><><><><><>
        #sched.step()

        # optional save
        if (epoch+1)%save_freq==0 and (COMM is None or COMM.rank==0):
            outname = os.path.join(outdir, "nety_ep%d.nn"%(epoch+1))
            torch.save(nety.state_dict(), outname)
            plt.savefig(outname.replace(".nn", "_train.png"))
            save_results_fig(outname,test_lab, test_pred) 
            
    # final save! 
    if COMM is None or COMM.rank==0:
        outname = os.path.join(outdir, "nety_epLast.nn")
        torch.save(nety.state_dict(), outname)
        plt.savefig(outname.replace(".nn", "_train.png"))
        save_results_fig(outname,test_lab, test_pred) 


if __name__ == "__main__":
    args = get_args()

    train_start_stop = test_start_stop = None
    if args.quickTest:
        train_start_stop = 2000, 2100
        test_start_stop = 0, 100
    if args.trainRange is not None:
        train_start_stop = args.trainRange
    if args.testRange is not None:
        test_start_stop = args.testRange
    do_training(args.input, args.labelName, args.imgsName, args.outdir,
                train_start_stop=train_start_stop,
                test_start_stop=test_start_stop,
                momentum=args.momentum,
                weight_decay=args.weightDecay, 
                nesterov=args.nesterov, damp=args.damp,
                dropout=args.dropout,
                lr=args.lr, bs=args.bs, ep=args.ep,
                arch=args.arch, loss=args.loss,
                logfile=args.logfile, loglevel=args.loglevel,
                display=not args.noDisplay, save_freq=args.saveFreq)

#   TODO
#   BINARY IMAGE CLASSIFIER -> get in the ballpark
#   Shell Image regressions -> fine tune using resolution shell
#   Spotfinding -> MultiHeadedAttenton models
