#!/bin/bash

DATASET_ORIGIN=../../videos/

ROOT=`pwd`/..
SRC=${ROOT}/new_dataset

echo $ROOT
echo ${DATASET_ORIGIN}

if [ ! -d ${SRC} ];
then
    mkdir ${SRC}
fi


for vid_path in $DATASET_ORIGIN/*
do
    vid_name=$(echo ${vid_path} | cut -f5 -d/ | cut -f1 -d.)
    echo "====================Running ${vid_name}===================="

    vid_src_path=${SRC}/${vid_name}

    if [ ! -d ${vid_src_path} ];
    then
        mkdir -p ${vid_src_path}/src
        cp ${vid_path} ${vid_src_path}/src/
        ffmpeg -y -i ${vid_src_path}/src/${vid_name}.mp4 -pix_fmt yuvj420p -vsync 0 -start_number 0 ${vid_src_path}/src/%010d.png
    fi
done
