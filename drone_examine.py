
import subprocess

name_format = 'drone_%d'

for i in [1,11,65,84]:
    name = name_format % i
    subprocess.run([
        'python',
        'examine.py',
        name,
        'results_' + name,
        'stats_' + name,
        'gt',
        '>' + name + '.txt'])
