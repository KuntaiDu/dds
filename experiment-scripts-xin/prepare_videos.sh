#!/bin/bash
src_dir_name=/new_dataset/out10fps720p/src
vid_name=out10fps720p #no suffix
qp=40
res=0.375
num_frames=661
ROOT=/data/yuanx

scale=480:270
echo $ROOT

image_src_path=${ROOT}/${src_dir_name}
vid_des_path=${ROOT}/${vid_name}_${res}_${qp}.mp4

ffmpeg -y \
-loglevel error \
-start_number 0 \
-i ${image_src_path}/%010d.png \
-vcodec libx264 \
-g 15 \
-keyint_min 15 \
-qp ${qp} \
-pix_fmt yuv420p \
-vf scale=${scale} \
-frames:v ${num_frames} \
${vid_des_path}

echo ${image_src_path}
