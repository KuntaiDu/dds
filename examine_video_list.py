
import subprocess
import yaml

with open('video_list.yaml', 'r') as f:
    video_list = yaml.load(f.read())

for video in video_list:
    result = subprocess.check_output([
        'python',
        'examine.py',
        f'{video}',
        f'results_{video}',
        f'stats_{video}',
        'gt'
    ])

    with open(f'{video}.txt', 'w') as f:
        f.write(result.decode('utf-8'))

    result = subprocess.check_output([
        'python',
        'examine_15frames.py',
        f'{video}',
        f'results_{video}',
        f'stats_{video}',
        'gt'
        ])
    with open(f'{video}_profile.txt', 'w') as f:
        f.write(result.decode('utf-8'))
