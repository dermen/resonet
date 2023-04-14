# coding: utf-8
import numpy as np
import glob
import re
import tabulate

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("inputs", type=str, nargs="+", help="wild cards for specifying .npz files")
args = parser.parse_args()

names = []
for glob_s in args.inputs:
    names += glob.glob(glob_s)

print(names)

def res_from_dirname(name):
    res = re.findall("[0-9]\.[0-9]+A", name)
    assert len(res)==1
    res = float(res[0].split("A")[0])
    return res

results = [] 
for n in names:
    d = np.load(n)
    s = d["result_string"][()]
    #print(s, "\n")
    top10 = float(s.split("perc=")[1].split("A")[0])
    mx = float(s.split("highest=")[1].split("A")[0])
    mn = float(s.split("Res:")[1].split("+")[0])
    sig = float(s.split("+-")[1].split()[0])
    dirname = s.split("Res:")[0]
    print(dirname)
    #from IPython import embed;embed()    
    target = res_from_dirname(dirname)
    factor = d['factor']

    results.append( ( target, mx,mn,sig,top10, d['pixsize'], d['detdist'],d[ 'wavelen']))
results = sorted(results)

# OLD WAY
#results = list(zip(*results))
#print(tabulate.tabulate(results))

# NEW WAY
header = ["label (A)", "highest", "mean", "stdev", "top10perc", "px(mm)", "dist(mm)", "wave(A)"] 
print(tabulate.tabulate(results, headers=header, floatfmt=".3f"))

