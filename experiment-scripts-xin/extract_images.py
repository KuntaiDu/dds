
import os
import subprocess

video_directory = r'/data/kuntai/aiskyeye_mp4/'
dest_directory = r'/data/kuntai/new_dataset/'
for video_name in os.listdir(video_directory):
    video_prefix = video_name.split('.')[0]
    retval = subprocess.call(['./extract_images.sh', video_directory + video_name, dest_directory + video_prefix + '/src/'])
    if retval != 0:
        print('%s failed' % video_name)
