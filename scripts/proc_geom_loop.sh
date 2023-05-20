#!/bin/bash

moredata=$1
model=$2
nproc=$3
arch=$4

tmps=resonettemp
find /mnt/tmpdata/data/ALL_DATA/ -maxdepth 1 -type d  -name "[0-9]*[0-9]A" > $tmps

find /mnt/tmpdata/data/ALL_DATA/ -maxdepth 1 -type l  -name "[0-9]*[0-9]A" >> $tmps

count=1
#for dirname in `find ${moredata} -maxdepth 1 -type d,l -name "[0-9]*[0-9]A"`
for dirname in `cat $tmps`
do
    # resolution target
    res=$(basename $dirname)

    # use a counter in filename in case of resolution duplicates
    outname=results.${count}.${res}

    # evaulate the model
    mpirun -n $nproc libtbx.python $MODZ/resonet/examples/proc_geom.py  $dirname $model ${outname} --predictor one_over_res \
    --geom --arch $arch

    count=$[$count +1]
done
