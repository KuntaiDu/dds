
import yaml
import subprocess

def load_and_write(conf):

    with open('dds_env.yaml', 'r') as f:
        dds_env = yaml.load(f.read())

    dds_env['dds_config'][0][10] = conf

    with open('dds_env.yaml', 'w') as f:
        f.write(yaml.dump(dds_env))


for conf in [1.6, 1.8, 1.9, 1.0]:
    load_and_write(conf)
    subprocess.run([
        'python',
        'workspace/all-videos.py',
        '0'
    ])
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
