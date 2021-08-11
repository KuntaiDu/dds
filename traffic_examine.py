
import subprocess

name_format = 'trafficcam_%d'

for i in [1,2,3,4,5,6,7]:
    name = name_format % i
    subprocess.run([
        'python',
        'new_examine.py',
        name,
        'results_' + name,
        'stats_' + name,
        'gt',
        '>' + name + '.txt'])
