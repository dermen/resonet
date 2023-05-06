#!/bin/bash

moredata=$1
model=$2
nproc=$3
arch=$4

count=1
for dirname in `find ${moredata} -maxdepth 1 -type d,l -name "[0-9]*[0-9]A"`
do
    # resolution target
    res=$(basename $dirname)

    # use a counter in filename in case of resolution duplicates
    outname=results.${count}.${res}

    # evaulate the model
    libtbx.python $MODZ/resonet/examples/proc_geom_joblib.py  $dirname $model ${outname} --predictor one_over_res \
    --geom --njobs $nproc --sanitize --arch $arch

    count=$[$count +1]
done
