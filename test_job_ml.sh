#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=1
#SBATCH -t 30
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -A m4326_g
#SBATCH -J debugs
#SBATCH --error=%J.err
#SBATCH --output=%J.out

source ~/dermen.sh

export master=$SCRATCH/train.6_1.45A_pt1/master.h5
export odir=$SCRATCH/temp.train.6

srun -N8 --tasks-per-node=1 --cpus-per-gpu=1 --gpus-per-node=4 python \
  10000 $master $odir --lr 6e-3 --bs 72 --arch res50 --loss L1 \
  --labelSel one_over_reso --momentum 0.9 --testRange 0 4200 \
  --trainRange 4200 42000 --useGeom --noDisplay --saveFreq 1



