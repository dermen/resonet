# usage
# $ python /home/teo/capstone-SLAC/resonet/scripts/plot_results_by_epoch.py "*.npz"

import numpy as np
import glob
import re
import tabulate
import pandas as pd
from matplotlib import pyplot as plt


class model:
    def __init__(self,name,num,res):
        self.name = str(name)
        self.res = str(res)
        self.num = str(num)
        self.epoch = int(self.name.split("ep")[1])
    
    def __str__(self):
        return "{0}.nn.{1}.{2}.npz".format(self.name, self.num, self.res)


def calculate_rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("inputs", type=str, nargs="+", help="wild cards for specifying .npz files")
args = parser.parse_args()

names = []
for glob_s in args.inputs:
    names += glob.glob(glob_s)


def res_from_dirname(name):
    res = re.findall("[0-9]\.[0-9]+A", name)
    assert len(res)==1
    res = float(res[0].split("A")[0])
    return res

results = [] 
mx_val = []
resolutions = ["1.42A",  "1.50A",  "1.62A",  "1.70A",  "1.74A",  "1.81A",  "1.8A",   "2.0A",   "2.80A",  "2.90A",
"1.45A",  "1.60A",  "1.66A",  "1.72A",  "1.76A",  "1.85A",  "1.95A",  "2.85A",  "5.45A"]

models = []
for n in names:
    ns = n.split(".")
    models += [model(ns[0], ns[2], ns[3]+"."+ns[4])]

rmse_by_ep = []
mad_by_ep = []
for e in range(5,215,5):
    rmses = []
    rmses2 = 0
    mad = 0
    for r in resolutions:
        n = ""
        for a in models:
             if ((str(e) == str(a.epoch)) and( str(r)  == str(a.res))):
                 n = str(a)
                 print(n)
        d = np.load(n)
        s = d["result_string"][()]
        top10 = float(s.split("perc=")[1].split("A")[0])
        mx = float(s.split("highest=")[1].split("A")[0])
        mn = float(s.split("Res:")[1].split("+")[0])
        sig = float(s.split("+-")[1].split()[0])
        dirname = s.split("Res:")[0]   
        target = res_from_dirname(dirname)
        factor = d['factor']
        res = d["res"][()]
        res_truth = np.ones(len(res))*int(target)
        rmse = calculate_rmse(res, res_truth)
        rmses.append(rmse)
        rmses2 += rmse ** 2
        results.append( ( target, mx,mn,sig,top10, d['pixsize'], d['detdist'],d[ 'wavelen'],rmse))
        mx_val.append(mx)
    rmse_by_ep.append(np.sqrt(rmses2))
    mad_by_ep.append(pd.Series(mx_val).mad())
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('trial.3')
ax1.plot(np.linspace(5,210,42), rmse_by_ep)
ax1.set_title("RMSE by Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("RMSE")
ax2.plot(np.linspace(5,210,42), mad_by_ep)
ax2.set_title("MAD by Epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MAD")
fig.savefig("trial.3.png")
fig.show()


#results = sorted(results)

# NEW WAY
header = ["label (A)", "highest", "mean", "stdev", "top10perc", "px(mm)", "dist(mm)", "wave(A)","RMSE"] 
print(tabulate.tabulate(results, headers=header, floatfmt=".3f"))
