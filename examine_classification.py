# USAGE
# python examine.py video_name results_fold stats_file gt
import sys
import os
import yaml
import pickle
import numpy
import torch
import torch.nn.functional as F
from pathlib import Path

with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())

video_name = sys.argv[1]
results_direc = sys.argv[2]
stats_file = sys.argv[3]
gt_mode = sys.argv[4]
# video_name = results_direc[11:]
print(video_name)
dirs = os.listdir(results_direc)

relevant_classes = dds_env['relevant_classes']


def parse_stats(stats_path):
    fname_to_size = {}
    # return total size
    with open(stats_path) as f:
        for cnt, line in enumerate(f):
            if cnt == 0:
                continue
            fields = line.split(',')
            fname = fields[0].split('/')[-1]
            total_size = int(float(fields[15])/1e4)
            fname_to_size[fname] = total_size
    return fname_to_size


fname_to_size = parse_stats(stats_file)

results_direc = Path(results_direc)

for name in dirs:
    # skip unwanted files
    if "req_regions" in name or "jpg" in name or "segment_size" in name or (results_direc / name).is_dir():
        continue

    with open(f'{results_direc / name}', 'rb') as f:
        target = pickle.load(f)

    with open(f'{results_direc / video_name}_gt', 'rb') as f:
        gt = pickle.load(f)

    assert target.keys() == gt.keys()

    mse = 0
    for key in target.keys():
        mse += F.mse_loss(torch.from_numpy(gt[key]), torch.from_numpy(target[key]))
    mse /= len(target.keys())
    print(f'{name} {fname_to_size[name]} {mse} 0')
