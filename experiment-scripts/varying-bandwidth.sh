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

    for low_threshold in {0.3, 0.35, 0.4, 0.45};
    do
        for threshold in {0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95};
        do
            python play_video.py --vid-name results/${vidname}_${low_threshold}_${threshold} --output-file stats --src ${images_dir} \
                   --low-results ${low_path} --high-results ${high_path} --ground-truth ${groundtruth} \
                   -lt ${low_threshold} -ht ${threshold} --intersection-threshold 0.3 --verbosity info
        done
    done

    for threshold in {0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95};
    do
        python play_video.py --vid-name results/${vidname}_0.5_${threshold} --output-file stats --src ${images_dir} \
               --low-results ${low_path} --high-results ${high_path} --ground-truth ${groundtruth} \
               -lt 0.5 -ht ${threshold} --intersection-threshold 0.3 --verbosity info
    done

    for threshold in {0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95};
    do
        python play_video.py --vid-name results/${vidname}_0.55_${$threshold} --output-file stats --src ${images_dir} \
               --low-results ${low_path} --high-results ${high_path} --ground-truth ${groundtruth} \
               -lt 0.55 -ht ${threshold} --intersection-threshold 0.3 --verbosity info
    done

    for threshold in {0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95};
    do
        python play_video.py --vid-name results/${vidname}_0.60_${$threshold} --output-file stats --src ${images_dir} \
               --low-results ${low_path} --high-results ${high_path} --ground-truth ${groundtruth} \
               -lt 0.60 -ht ${threshold} --intersection-threshold 0.3 --verbosity info
    done

    for threshold in {0.7, 0.75, 0.8, 0.85, 0.9, 0.95};
    do
        python play_video.py --vid-name results/${vidname}_0.65_${$threshold} --output-file stats --src ${images_dir} \
               --low-results ${low_path} --high-results ${high_path} --ground-truth ${groundtruth} \
               -lt 0.65 -ht ${threshold} --intersection-threshold 0.3 --verbosity info
    done

    for threshold in {0.75, 0.8, 0.85, 0.9, 0.95};
    do
        python play_video.py --vid-name results/${vidname}_0.70_${$threshold} --output-file stats --src ${images_dir} \
               --low-results ${low_path} --high-results ${high_path} --ground-truth ${groundtruth} \
               -lt 0.70 -ht ${threshold} --intersection-threshold 0.3 --verbosity info
    done

done
