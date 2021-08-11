
import subprocess
import yaml

lst = [14,15,16,17,18,19,20,21,22,23,24,29,30]

for i in lst:

    j = i

    subprocess.run([
        'python',
        'examine.py',
        f'drone_{j}',
        f'results_drone_{j}',
        f'stats_drone_{j}',
        'gt'
    ])
