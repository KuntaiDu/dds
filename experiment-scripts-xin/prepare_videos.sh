#!/bin/bash
src_dir_name=$1
vid_name=$2 #no suffix
qp=$3
res=$4
num_frames=$5
ROOT=/data/yuanx

if [ "$res" = "low" ]; then
  scale=256:144
else
  scale=896:504
fi

echo $ROOT

image_src_path=${ROOT}/${src_dir_name}
vid_des_path=${ROOT}/${vid_name}_${qp}_${res}.mp4

# mkdir -p ${image_des_path}

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
