
import os
import sys
import subprocess
from pathlib import Path

import yaml

with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())

_, vid_name, res, qp, num_frames, scale = sys.argv

dataset_path = Path(dds_env['dataset'])

images_src_path = dataset_path / f'{vid_name}' / 'src'
vid_des_path = dataset_path / f'{vid_name}_{res}_{qp}.mp4'
images_des_path = dataset_path / f'{vid_name}_{res}_{qp}' / 'src'


subprocess.run([
    'ffmpeg',
    '-y',
    '-loglevel', 'error',
    '-start_number', '0',
    '-i', f'{images_src_path}/%010d.png',
    '-vcodec', 'libx264',
    '-g', '15',
    '-keyint_min', '15',
    '-qp', f'{qp}',
    '-vf', f'scale={scale}',
    '-pix_fmt', 'yuv420p',
    '-frames:v', f'{num_frames}',
    f'{vid_des_path}'
])

os.system(f'mkdir -p {images_des_path}')

subprocess.run([
    'ffmpeg',
    '-i', f'{vid_des_path}',
    '-vsync', '0',
    '-start_number', '0',
    f'{images_des_path}/%010d.png'
])
