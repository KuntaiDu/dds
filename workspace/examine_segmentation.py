# USAGE
# python examine.py video_name results_fold stats_file gt
from skimage import measure
import sys
import os
import yaml
import pickle
import numpy
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
sys.path.append('../')
import dds_utils

#thresh_low = 0.001
#thresh_high = 0.01
#thresh_low_gt = 0.001
#thresh_high_gt = 0.01
thresh_low, thresh_high, thresh_low_gt, thresh_high_gt = 0, 0.004, 0, 0.004


def process(x, is_gt=False):

    low, high = thresh_low, thresh_high
    # cast to int to prevent overflow
    x = torch.tensor(measure.label(x.numpy().astype(int))).cuda()
    nclass = torch.max(x).item()

    mask = torch.zeros_like(x).cuda()
    # cleaning
    for i in range(1, nclass + 1):
        size = torch.sum(x == i) * 1.0 / x.numel()
        if size < low or size > high:
            x[x == i] = 0
            mask[x == i] = 1
    # relabel to binary
    x[x != 0] = 1
    return x.cpu(), mask.cpu()



video_name = sys.argv[1]
results_direc = sys.argv[2]
stats_file = sys.argv[3]
gt_mode = sys.argv[4]
# video_name = results_direc[11:]
# print(video_name)
dirs = os.listdir(results_direc)


def parse_stats(stats_path):
    fname_to_size = {}
    # return total size
    with open(stats_path) as f:
        for cnt, line in enumerate(f):
            if cnt == 0:
                continue
            fields = line.split(',')
            fname = fields[0].split('/')[-1]
            total_size = int(float(fields[15]))
            fname_to_size[fname] = total_size
    return fname_to_size


fname_to_size = parse_stats(stats_file)
results_direc = Path(results_direc)

gt_file = results_direc / f'{video_name}_mpeg_1.0_26'
if not gt_file.exists():
    gt_file = results_direc / f'{video_name}_gt'


with open(f'{gt_file}', 'rb') as f:
    gt = pickle.load(f)
#    import pdb; pdb.set_trace()
for key in gt.keys():
    gt[key] = process(gt[key], True)


for name in dirs:
    # skip unwanted files
    if "req_regions" in name or "jpg" in name or "segment_size" in name or (results_direc / name).is_dir() or 'region_proposal' in name or 'final' in name:
        continue

    # if "0.25_1" not in name:
    #	continue
    if 'mpeg' not in name and 'dds' not in name and 'cloudseg' not in name:
        #if 'dds_1.0_36' not in name:
        continue

#    if 'dds' not in name:
#        continue

    print(f'Processing {name}', file=sys.stderr)

    with open(f'{results_direc / name}', 'rb') as f:
        x = pickle.load(f)

    assert x.keys() == gt.keys()

    F1 = []
    sameshape = True
    for key in x.keys():
        #import pdb; pdb.set_trace()

        x_ind, x_mask = process(x[key])
        gt_ind, gt_mask = gt[key]
        x_ind, x_mask, gt_ind, gt_mask = x_ind.cuda(
        ), x_mask.cuda(), gt_ind.cuda(), gt_mask.cuda()

        if gt_ind.shape != x_ind.shape:
            sameshape = False
            break

        # clear large objects, mutual version
        # x_ind[(x_mask + gt_mask) > 0] = 0
        # gt_ind[(x_mask + gt_mask) > 0] = 0

        # skip background class
        background = torch.zeros_like(gt_ind)

        # filter out all background pixels
        correct_pixels = torch.where((gt_ind == x_ind) & ((gt_ind != background) | (
            x_ind != background)), torch.ones_like(gt_ind), torch.zeros_like(gt_ind))
        total_pixels = torch.where(((gt_ind != background) | (
            x_ind != background)), torch.ones_like(gt_ind), torch.zeros_like(gt_ind))

        ncorrect = len(correct_pixels.nonzero())
        ntotal = len(total_pixels.nonzero())
        if ntotal != 0:
            F1.append((ncorrect)/(ntotal))

    if sameshape:
        try:
            print(f'{name} {fname_to_size[name]} {np.mean(F1)} 0')
        except KeyError:
            continue
        # print(f'{name} {fname_to_size[name]} {np.mean(F1)} 0', file=sys.stderr)
