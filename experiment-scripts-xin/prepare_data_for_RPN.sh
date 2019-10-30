#!/bin/bash
vid_name=$1
res=$2
qp=$3
num_frames=$4
scale=$5
src_dir_name=new_dataset/${vid_name}/src/
des_dir_name=new_dataset/${vid_name}_${res}_${qp}/src
ROOT=/data2/kuntai-gpuc/kuntai

echo $ROOT

image_src_path=${ROOT}/${src_dir_name}
vid_des_path=${ROOT}/${vid_name}_${res}_${qp}.mp4

/usr/bin/ffmpeg -y \
-loglevel error \
-start_number 0 \
-i ${image_src_path}/%010d.png \
-vcodec libx264 \
-g 15 \
-keyint_min 15 \
-qp ${qp} \
-vf scale=${scale} \
-pix_fmt yuv420p \
-frames:v ${num_frames} \
${vid_des_path}

image_des_path=${ROOT}/${des_dir_name}
mkdir -p ${image_des_path}

/usr/bin/ffmpeg -y \
-i ${vid_des_path} \
-vsync 0 \
-start_number 0 \
${image_des_path}/%010d.png
