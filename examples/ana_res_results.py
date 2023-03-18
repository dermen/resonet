import sys
import numpy as np
import pylab as plt

results_file = sys.argv[1]
dat = np.load(results_file)

# we assume the shots filenames end with *_%05d.cbf, e.g. something_00001.cbf
shot_num = [int(f.split("_")[-1].split(".")[0]) for f in dat['fignames']]
order = np.argsort(shot_num)
res = dat['res']

mean_res = np.mean(res)
std_res = np.std(res)
median_res = np.median(res)
high_res = np.min(res)

target = dat['target_res']
plt.hlines(dat['target_res'], 0, len(dat['rads']), color='tomato', ls='--', lw=2,label="%.2fA"% target)
plt.plot( res[order], '.')
plt.grid(1)
plt.gca().tick_params(labelsize=16)
plt.ylabel("Resolution ($\AA$)", fontsize=18)
plt.xlabel("Shot #", fontsize=18)
plt.legend(prop={"size":16})
plt.suptitle("Datafile: %s. Mean res.: %.2fA +- %.2fA\nMedian res.: %.2fA, Highest res.: %.2fA" 
        % (results_file,mean_res, std_res, median_res, high_res), fontsize=18)

plt.gcf().set_size_inches((8,4))
plt.subplots_adjust(left=.15, bottom=.16, top=.84)

plt.show()

