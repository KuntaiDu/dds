#! /bin/bash

DATASET_ORIGIN=../new_dataset
OUTPUT_FILE=stats

if [ ! -d "results" ];
then
    mkdir results
fi


mode=$1
video=$2
original_images=${DATASET_ORIGIN}/${video}/src

if [ $mode == "gt" ];
then
    vname=$3
    qp=$4
    python play_video.py --vid-name results/${vname} --high-src ${original_images} --resolutions 1.0 \
           --output-file ${OUTPUT_FILE} --max-size 0.3 --low-threshold 0.3 --high-threshold 0.8 \
           --enforce-iframes --qp ${qp} --verbosity info
elif [ $mode == "mpeg" ];
then
    vname=$3
    qp=$4
    res=$5
    python play_video.py --vid-name results/${vname} --resolutions ${res} \
           --high-src ${original_images} --output-file ${OUTPUT_FILE} --ground-truth results/${video}_gt \
           --max-size 0.3 --low-threshold 0.3 --high-threshold 0.8 --enforce-iframes --qp ${qp} --verbosity info
elif [ $mode == "dds" ];
then
    vid_name=$2
    vname=$3
    low_qp=$4
    high_qp=$5
    low=$6
    high=$7
    low_results=results/${video}_mpeg_${low}_${low_qp}
    python play_video.py --vid-name results/${vname} --high-src ${original_images} \
           --resolutions ${low} ${high} --low-results ${low_results} --output-file ${OUTPUT_FILE} \
           --ground-truth results/${video}_gt --max-size 0.3 \
           --low-threshold 0.3 --high-threshold 0.8 --enforce-iframes --qp ${low_qp} ${high_qp} --verbosity info
fi
