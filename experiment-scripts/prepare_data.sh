#!/bin/bash

DATASET_ORIGIN=$1

ROOT=/data/yuanx/
SRC=${ROOT}/new_dataset

echo $ROOT

if [ ! -d ${SRC} ];
then
    mkdir ${SRC}
fi

for vid_path in $DATASET_ORIGIN/*.mp4
do
    echo $vid_path
    vid_name=$(echo ${vid_path} | cut -f4 -d/ | cut -f1 -d.)
    vid_src_path=${SRC}/${vid_name}

    if [ ! -d ${vid_src_path} ];
    then
        mkdir -p ${vid_src_path}/src
        cp ${vid_path} ${vid_src_path}/src/
        ffmpeg -y -i ${vid_path} -pix_fmt yuvj420p -vsync 0 -start_number 0 ${vid_src_path}/src/%010d.png
    fi

done
