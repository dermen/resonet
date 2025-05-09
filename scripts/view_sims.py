
import textwrap
import pylab as plt
import h5py
import numpy as np
import os
from argparse import ArgumentParser

import matplotlib
from matplotlib.widgets import Slider
from matplotlib.pyplot import waitforbuttonpress


def get_args():
    ap = ArgumentParser()
    ap.add_argument("input", help="input hdf5 file created by resonet", type=str)
    args = ap.parse_args()
    return args


def main():
    matplotlib.rcParams['keymap.back'].remove('left')
    matplotlib.rcParams['keymap.forward'].remove('right')
    matplotlib.use("TkAgg")
    args = get_args()

    h = h5py.File(args.input, "r")
    imgs = h['images']
    labs = h['labels']
    peaks = None
    if "peak_segments" in h.keys():
        peaks = h['peak_segments']

    def get_label_s(lab_val):
        if isinstance(lab_val, float) or isinstance(lab_val, np.float32):
            if np.isnan(lab_val):
                val_s = "nan"
            elif lab_val == int(lab_val):
                val_s = "%d" % lab_val
            else:
                val_s = "%.4f" % lab_val
        else:
            val_s = str(lab_val)
        return val_s

    def print_label_info(i_img):
        label_string = ""
        for lab_val, name in zip(labs[i_img], labs.attrs['names']):
            if name == "pdb":
                if not np.isnan(lab_val):
                    lab_val = os.path.basename(labs.attrs['pdbmap'][int(lab_val)])
                else:
                    lab_val = "nan"
            val_s = get_label_s(lab_val)
            label_string += "%s=%s;  " % (name, val_s)

        for geom_val, name in zip(h['geom'][i_img], h['geom'].attrs['names']):
            val_s = get_label_s(geom_val)
            label_string += "%s=%s " % (name, val_s)
        return label_string

    n_imgs = imgs.shape[0]
    assert n_imgs > 0
    img = imgs[0]

    fig = plt.figure(1)
    fig.set_size_inches(7,4.*7/10)
    fig.i_img = 0
    fig.n_imgs = n_imgs
    fig.use_mask = False
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, width_ratios=[1,30,15], left=0.1, wspace=0.2,  right=0.9)
    axcmap = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    axtext = fig.add_subplot(gs[2])
    #ax = fig.add_axes([0.12, 0.1, 0.5, 0.8])
    #axcmap = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
    #axtext = fig.add_axes([0.5, 0.05, 0.5, 0.9])

    cm = plt.cm.gray_r.with_extremes(bad='r')
    ax.imshow(img, vmin=0, vmax=255, cmap=cm)

    amp_slider = Slider(
        ax=axcmap,
        label="Colorscale",
        valmin=0,
        valmax=255,
        valinit=img.mean()+3.5*img.std(),
        orientation="vertical"
    )

    # The function to be called anytime a slider's value changes
    def update_clim(val):
        ax.images[0].set_clim(0,val)
        fig.canvas.draw_idle()
    amp_slider.on_changed(update_clim)
    update_clim(amp_slider.valinit)

    def check_exit():
        if not plt.fignum_exists(1):
            exit()

    def press(event):
        if event.key == 'right':
            fig.i_img += 1
        elif event.key == "left":
            fig.i_img = fig.i_img - 1
        elif event.key == 'up':
            fig.use_mask = not fig.use_mask
            print("Toggling mask", fig.use_mask)

        fig.i_img = max(fig.i_img, 0)
        fig.i_img = min(fig.i_img, fig.n_imgs - 1)

        if event.key=="escape":
            fig.i_img = fig.n_imgs

    fig.canvas.mpl_connect('key_press_event', press)

    axtext.text(0.5,0.5, "label", fontsize=8, transform=axtext.transAxes, ha="center", va="center")
    axtext.axis('off')

    while fig.i_img < n_imgs:
        label_s = print_label_info(fig.i_img)
        label_s = "\n".join(textwrap.wrap(label_s, width=36, break_long_words=False))
        axtext.texts[0].set_text("Labels:\n"+label_s)
        img = imgs[fig.i_img].copy().astype(np.float32)
        if peaks is not None and fig.use_mask:
            in_peak = peaks[fig.i_img]
            img[in_peak] = np.nan
        ax.images[0].set_data(img)
        ax.set_title("image %d/%d (Arrow Right / Left to scroll)" % (fig.i_img+1, fig.n_imgs), fontsize=10)
        if waitforbuttonpress():
            continue
        else:
            # check if user clicked window close button
            check_exit()

    plt.close()


if __name__=="__main__":
    main()
