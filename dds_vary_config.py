
import yaml
import subprocess

def load_and_write(qp):

    with open('dds_env.yaml', 'r') as f:
        dds_env = yaml.load(f.read())

    dds_env['dds_config'][0][5] = qp

    with open('dds_env.yaml', 'w') as f:
        f.write(yaml.dump(dds_env))


for low_config in [60, 90, 120]:
    load_and_write(low_config)
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
