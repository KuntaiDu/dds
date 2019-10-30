
import subprocess

for i in range(100):
    if i < 17:
        continue
    subprocess.run([
        'python',
        'workspace/all-videos.py',
        '0',
        f'{i}'
        ])

    subprocess.run([
        'python',
        'workspace/all-videos.py',
        '1',
        f'{i}'
        ])

    subprocess.run([
        'python',
        'workspace/all-videos.py',
        '2',
        f'{i}'
        ])
