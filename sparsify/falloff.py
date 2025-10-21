from mpi4py import MPI
COMM = MPI.COMM_WORLD
import numpy as np
import find_spots
import fabio
from scipy.ndimage import binary_dilation
from dials.command_line.find_spots import phil_scope
import glob

params = phil_scope.extract()
params.spotfinder.threshold.algorithm="dispersion"
params.spotfinder.threshold.dispersion.global_threshold=400
params.spotfinder.threshold.dispersion.kernel_size=[6,6]
params.spotfinder.threshold.dispersion.sigma_strong=4
params.spotfinder.filter.max_spot_size=15
params.spotfinder.filter.min_spot_size=3

#fnames = glob.glob("../A6collect_15MGy/A6_1_*cbf")
fnames = glob.glob("../A7collect_30MGy/A7_1_*cbf")
fnames = sorted(fnames, key=lambda x: int(x.split("_")[-1].split(".")[0]) )
mask = np.load("ring_mask.npy")
mask = binary_dilation(mask, iterations=2)

x=1231.5
y=1263.5
img_shape = 2527, 2463
Y,X = np.indices(img_shape)
all_vals = []
all_inds = []
R = np.sqrt((X-x)**2 + (Y-y)**2)
bins = np.arange(500)
bin_cent = 0.5*(bins[1:] + bins[:-1])
#norm = np.histogram(R, bins=bins)[0]

all_f = []
for i, f in enumerate(fnames[:]):
    if i % COMM.size != COMM.rank:
        continue
    print(f)
    img = fabio.open(f).data
    is_neg = img == -1
    is_neg = binary_dilation(is_neg, iterations=2)
    mask = np.logical_and(mask, ~is_neg)
    spots = find_spots.dials_find_spots(img*mask, params)
    rvals = R[spots]
    pixvals = (img*mask)[spots]
    vals = np.histogram(rvals, bins=bins, weights=pixvals)[0]
    norm = np.histogram(rvals, bins=bins)[0]
    
    all_vals.append(np.nan_to_num(vals/norm))
    all_inds.append(i)
    all_f.append(f)

all_vals = COMM.reduce(all_vals)
all_f = COMM.reduce(all_f)
all_inds = COMM.reduce(all_inds)
if COMM.rank==0:
    print("saving")
    vals = np.array(all_vals)
    inds = np.array(all_inds)
    order = np.argsort(inds)
    vals = vals[order]
    inds = inds[order]
    f = np.array(all_f)[order]
    np.savez("stats_A7_3", vals=vals, inds=inds, f=f, bins=bins)
    
    exit()
    def statfunc(dat):
        dat = dat[dat > 0]
        if not dat.size:
            return 0
        else:
            return dat.mean()

    bin_cent = 0.5*(bins[:-1] + bins[1:])
    nbin=20
    #t = 8.9247000
    t = 21.7329000
    stats = binned_statistic(bin_cent, vals, bins=nbin, statistic= statfunc)[0]
    imshow(stats+1e-6,interpolation="none");gca().set_aspect("auto")
    stat_bin = [b.mean() for b in np.array_split(bin_cent, nbin)]

    gca().tick_params(labelsize=12)
    gca().set_xticks(arange(0,nbin,3))
    gca().set_yticks(arange(0,len(stats),500))
    gca().set_yticklabels(["%.1f"%(y*t/3600) for y in arange(0,4000,500)])
    gca().set_xticklabels(["%.1f$^o$"%(np.arctan(stat_bin[i]*0.172/160)*180/np.pi) for i in gca().get_xticks()])
    xlabel("scattering angle (deg.)", fontsize=16)
    ylabel("exposure time (hours)",fontsize=16 )
    colorbar()
    im = gca().images[-1]
    im.colorbar.ax.set_title("mean Bragg intensity", pad=12)
    im.set_norm(mpl.colors.LogNorm(vmin=100,vmax=10000))
    im.colorbar.ax.tick_params(labelsize=12)
    im.set_cmap("gnuplot")
    title("A6collect_15MGy", fontsize=18)
        

