
import subprocess


name_list_drone = ['drone_%d' % i for i in [14,15,16,17,18,19,20,21,22,23,24,29,30]]

for name in name_list_drone:
    subprocess.run([
        'python',
        'new_examine.py',
        name,
        'results_' + name,
        'stats_' + name,
        'gt',
        '>' + name + '.txt'])
