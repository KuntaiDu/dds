
import yaml


def load():

    with open('dds_env.yaml', 'r') as f:
        dds_env = yaml.load(f.read())
    with open(dds_env['video_list'], 'r') as f:
        video_list = yaml.load(f.read())

    mpeg_configs = {}
    dds_config = {}
    gt_config = {}
    names = video_list

    for name in names:
        mpeg_configs[name] = dds_env['mpeg_configs']
        dds_config[name] = dds_env['dds_config']
        gt_config[name] = dds_env['gt_config']

    return names, mpeg_configs, dds_config, gt_config
