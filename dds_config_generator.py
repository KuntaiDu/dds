
import yaml


with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())
with open(dds_env['video_list'], 'r') as f:
    video_list = yaml.load(f.read())

def load():

    mpeg_configs = {}
    dds_config = {}
    gt_config = {}
    names = video_list

    for name in names:
        mpeg_configs[name] = [[0.8, 36], [0.8, 26]]
        dds_config[name] = [[0.8,36,0.8,26,0.,15,'no_filter_combined_bboxes',0.5, 0., 0.01]]
        gt_config[name] = [1.0, 24]

    return names, mpeg_configs, dds_config, gt_config
