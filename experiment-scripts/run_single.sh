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
           --max-size 0.3 --low-threshold 0.3 --high-threshold 0.3 --enforce-iframes --qp ${qp} --verbosity info
elif [ $mode == "dds" ];
then
    vname=$3
    low_qp=$4
    high_qp=$5
    low=$6
    high=$7
    rpn_enlarge_ratio=$8
    batch_size=$9
    prune_score=$10
    objfilter_iou=$11
    size_obj=$12
    low_results=results/${video}_mpeg_${low}_${low_qp}
    python play_video.py --vid-name ${RESULTS_DIR}/${vname} --high-src ${original_images} \
           --resolutions ${low} ${high} --low-results ${low_results} --output-file ${OUTPUT_FILE} \
           --ground-truth results/${video}_gt --max-size 0.3 \
           --low-threshold 0.3 --high-threshold 0.3 --enforce-iframes \
           --batch-size ${batch_size} --prune-score ${prune_score} \
           --objfilter-iou ${objfilter_iou} --size-obj ${size_obj} \
           --qp ${low_qp} ${high_qp} --verbosity info --rpn_enlarge_ratio ${rpn_enlarge_ratio} \

 fi
