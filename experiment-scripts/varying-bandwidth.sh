#! /bin/bash

EMULATION_RESULTS_DIR=$1

for vid_direc in ${EMULATION_RESULTS_DIR}/*;
do
    if [ ! -d $vid_direc ];
    then
        continue
    fi

    IFS='/' read -ra VID <<< "${vid_direc}"
    vidname=${VID[3]}

    low_path=${vid_direc}/MPEG/23_0.375/Results
    high_path=${vid_direc}/MPEG/23_0.75/Results
    groundtruth=${vid_direc}/GroundTruth/Results
    images_dir=${vid_direc}/0.375

    if [ ! -d results ];
    then
       mkdir results
    fi

    for threshold in {0.1,0.2,0.3,0.4,0.5,0.6,0.7};
    do
        python play_video.py --vid-name results/${vidname} --output-file stats --src ${images_dir} \
               --low-results ${low_path} --high-results ${high_path} --ground-truth ${groundtruth} \
               -lt ${threshold} -ht 0.8 --verbosity info
    done
done
