#!/bin/bash

vid_name=$1
des_dir_name=$2


vid_src_path=${vid_name}
image_des_path=${des_dir_name}

mkdir -p ${image_des_path}

/usr/bin/ffmpeg -y \
-i ${vid_src_path} \
-vsync 0 \
-start_number 0 \
${image_des_path}/%010d.png

# -vcodec mjpeg \
# -pix_fmt yuvj420p \
# -g 8 \
# -q:v 2 \
