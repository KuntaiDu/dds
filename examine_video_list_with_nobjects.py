
import subprocess
import yaml

with open('video_list.yaml', 'r') as f:
    video_list = yaml.load(f.read())

for video in video_list:
    result = subprocess.check_output([
        'python',
        'examine_with_nobjects.py',
        f'{video}',
        f'results_{video}',
        f'stats_{video}',
        'gt'
    ])

    with open(f'{video}_nobjects.yaml', 'w') as f:
        f.write(result.decode('utf-8'))
