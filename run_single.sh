#!/bin/bash

EMULATION_RESULTS_DIR="../new_dataset"
OUTPUT_FILE="stats"

vidname=drivers_0_00_00_00
vid_direc=${EMULATION_RESULTS_DIR}/${vidname}/

low=0.2
high=0.375

mkdir results

groundtruth=${vid_direc}/GroundTruth/Results
images_dir=${vid_direc}/src/${low}
original_images=${vid_direc}/src/
low_results=results/${vidname}_mpeg_${low}

python play_video.py --vid-name results/test --low-src ${images_dir} --high-src ${original_images} --resolutions ${low} ${high} \
       --output-file ${OUTPUT_FILE} --max-size 0.3 --low-threshold 0.3 --high-threshold 0.8 --debug-mode --verbosity info

# python play_video.py --vid-name results/${vidname}_${low}_${high} --low-src ${images_dir} --high-src ${original_images} --resolutions ${high} \
#        --output-file ${OUTPUT_FILE} --low-results ${low_path} --max-size 0.8 --low-threshold 0.3 --high-threshold 0.8 --debug-mode --verbosity info
