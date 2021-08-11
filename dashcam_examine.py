
import subprocess

for i in range(9):

    j = i + 1

    subprocess.run([
        'python',
        'examine.py',
        f'dashcam_{j}',
        f'results_dashcam_{j}',
        f'stats_dashcam_{j}',
        'gt'
    ])
