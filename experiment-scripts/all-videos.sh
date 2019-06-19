#! /bin/bash

DATASET_ORIGIN=../scratch
OUTPUT_FILE=stats

if [ ! -d "results" ];
then
    mkdir results
fi

echo "Running Groundtruth and MPEG"
for vid_path in $DATASET_ORIGIN/*
do
    vid_name=$(echo ${vid_path} | cut -f2 -d/)

    # Ground truth
    original_images=${vid_path}/src/
    python play_video.py --vid-name results/${vidname}_gt --low-src ${original_images} --high-src ${original_images} --resolutions 1.0 \
           --output-file ${OUTPUT_FILE} --max-size 0.8 --low-threshold 0.3 --high-threshold 0.8 --verbosity info

    # Run MPEG
    for resolution in {0.1,0.15,0.2,0.25,0.375,0.5,0.625,0.75,0.875};
    do
        images_direc=${vid_path}/src/${resolution}/
        python play_video.py --vid-name results/${vidname}_DDS_${resolution} --low-src ${images_direc} --resolutions ${resolution} \
               --high-src ${original_images} --output-file ${OUTPUT_FILE} --ground-truth results/${vidname}_gt --max-size 0.8 \
               --low-threshold 0.3 --high-threshold 0.8 --verbosity info
    done

done

echo "Running DDS"
for vid_path in $DATASET_ORIGIN/*
do
    vid_name=$(echo ${vid_path} | cut -f2 -d/)
    original_images=${vid_path}/src/
    # Run DDS
    low=0.2
    for high in {0.375,0.5};
    do
        low_images=${vid_path}/src/${low}/
        python play_video.py --vid-name results/${vidname}_DDS_${resolution} --low-src ${low_images} --high-src ${original_images} \
               --resolutions ${low} ${high} --output-file ${OUTPUT_FILE} --ground-truth results/${vidname}_gt --max-size 1.0 \
               --low-threshold 0.3 --high-threshold 0.8 --verbosity info
    done
done
