#!/bin/bash

model=$1

count=1
for dirname in `find /data/blstaff/SOLTIS/AI_PREDICTION/  -maxdepth 2 -type d -name "[0-9]*[0-9]A"`
do
    # resolution target
    res=$(basename $dirname)

    # use a counter in filename in case of resolution duplicates
    outname=results.${count}.${res}

    # evaulate the model
    mpirun -n 24 python $MODZ/resonet/examples/proc_geom.py  $dirname $model ${outname} --predictor one_over_res --geom 

    count=$[$count +1]
done
