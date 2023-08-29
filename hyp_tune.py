from isashumod import spotFinder
import random
import numpy as np
import sys
import time


outdir = sys.argv[1]
num_ep = int(sys.argv[2])
gpu_num = int(sys.argv[3])

loaderRoot = "/mnt/tmpdata/data/isashu/newLoaders/threeMax/smallerLoaders/"
loaderUnseenRoot = "/mnt/tmpdata/data/isashu/newLoaders/threeMax/smallerLoaders/"

def gen_params():
    num = 18  #random.choice([18, 34, 50])
    optim = 0
    lr = 4e-6 #np.exp(np.random.uniform(np.log(1e-10), np.log(1e-3)))
    #mom = random.choice([0, 0.2, 0.5, 0.8, 0.99])
    mom = 0.99 #random.choice([0, 0.9, 0.99])
    wd = 0#np.exp(np.random.uniform(np.log(1e-3), np.log(1e-1)))
    two_fc_mode = 0#random.choice([0,1])

    params = {
        'num': num,
        'optim': optim,
        'lr': lr,
        'mom': mom,
        'two_fc_mode': bool(two_fc_mode),
        'wd': wd
    }

    return params


while True:
    params = gen_params()
    print(params)
    try:
        spotFinder.main(outdir, loaderRoot, loaderUnseenRoot, gpu_num=gpu_num, epochs=num_ep, **params)
    except Exception as err:
        print(str(err))
        pass
    print("waiting 10 sec for GPU to clear... ") 
    time.sleep(10)

'''
For resnet-18:
learning rate should be between 2e-6 and 9e-6
momentum should be between 0.8 and 0.99

I don't know why it always does the final save in the first run folder created by hyp_tune
'''