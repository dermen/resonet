#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 60:00:00
#SBATCH -p short
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=aleksandercichosz@ucsb.edu
#SBATCH --mail-type=start,end
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
source ~/setup_cuda.sh
source ~/dials.sh

srun python net.py 150 baxter_master.h5 test.baxter --lr 0.00026366508987303583 --bs 8 --weightDecay 0.0001519911082952933 --saveFreq 5 --arch res34 --labelName multi --loss BCE2 --testRange 0 100 --trainRange 100 1000 --momentum 0.9829450917426307