#!/bin/bash

moredata=$1
model=$2
arch=$3

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
    mpirun -n 8 libtbx.python $MODZ/resonet/scripts/eval_reso.py  $dirname $model ${outname} \
    --arch $arch  --quads A B C D --gpus --leaveOnGpu

    count=$[$count +1]
done
