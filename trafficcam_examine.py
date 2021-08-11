
import subprocess

for i in range(7):

    j = i + 1

    subprocess.run([
        'python',
        'examine.py',
        f'trafficcam_{j}',
        f'results_trafficcam_{j}',
        f'stats_trafficcam_{j}',
        'gt'
    ])
