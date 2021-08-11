
import subprocess

name_format = 'dashcam_%d'

for i in [1,2,3,4,5,6,7,8,9]:
    name = name_format % i
    subprocess.run([
        'python',
        'examine.py',
        name,
        'results_' + name,
        'stats_' + name,
        'gt'])
