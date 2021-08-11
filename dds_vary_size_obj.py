
import yaml
import subprocess

def load_and_write(size):

    with open('dds_env.yaml', 'r') as f:
        dds_env = yaml.load(f.read())

    dds_env['dds_config'][0][9] = size

    with open('dds_env.yaml', 'w') as f:
        f.write(yaml.dump(dds_env))


for size in [0.04, 0.01, 0.0025, 0.000625]:
    load_and_write(size)
    #subprocess.run([
    #    'python',
    #    'workspace/all-videos.py',
    #    '0'
    #])
    subprocess.run([
        'python',
        'workspace/all-videos.py',
        '1'
    ])
    subprocess.run([
        'python',
        'workspace/all-videos.py',
        '2'
    ])
