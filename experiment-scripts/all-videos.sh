#! /bin/bash

DATASET_ORIGIN=../new_dataset
OUTPUT_FILE=stats

if [ ! -d "results" ];
then
    mkdir results
fi

echo "Running Groundtruth and MPEG"
for vid_path in $DATASET_ORIGIN/*
do
    vid_name=$(echo ${vid_path} | cut -f3 -d/)
    original_images=${vid_path}/src/

    # Ground truth
    python play_video.py --vid-name results/${vid_name}_gt --low-src ${original_images} --high-src ${original_images} --resolutions 1.0 \
           --output-file ${OUTPUT_FILE} --max-size 0.3 --low-threshold 0.3 --high-threshold 0.8 --enforce-iframes --verbosity info

    # Run MPEG
    for res in {0.15,0.2,0.25,0.375,0.5,0.625,0.75};
    do
        images_direc=${vid_path}/src/${res}/
        python play_video.py --vid-name results/${vid_name}_mpeg_${res} --low-src ${images_direc} --resolutions ${res} \
               --high-src ${original_images} --output-file ${OUTPUT_FILE} --ground-truth results/${vid_name}_gt --max-size 0.3 \
               --low-threshold 0.3 --high-threshold 0.8 --enforce-iframes --verbosity info
    done

done

echo "Running DDS"
for vid_path in $DATASET_ORIGIN/*
do
    vid_name=$(echo ${vid_path} | cut -f3 -d/)
    original_images=${vid_path}/src/
    # Run DDS
    low=0.2
    for high in {0.375,0.5};
    do
        low_images=${vid_path}/src/${low}
        low_results=results/${vid_name}_mpeg_${low}
        python play_video.py --vid-name results/${vid_name}_dds_${low}_${high} --low-src ${low_images} --high-src ${original_images} \
               --resolutions ${low} ${high} --low-results ${low_results} --output-file ${OUTPUT_FILE} --ground-truth results/${vid_name}_gt \
               --max-size 0.3 --low-threshold 0.3 --high-threshold 0.8 --enforce-iframes --verbosity info
    done
done
