import numpy as np
import matplotlib.pyplot as plt

with open('stats_new_gt', "r") as f:
    lines = f.readlines()
    f.close()

f1_list = []
bw_list = []
notation_list = []

import os
import re
import logging
import argparse
from dds_utils import *
# Evaluation and writing results
from merger import *
# Read Groundtruth results

import os


data_dir = "/data/yuanx/new_dataset"
# mpeg_resolutions = [0.15, 0.2, 0.25, 0.375, 0.5, 0.625, 0.75]
mpeg_resolutions = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.375, 0.35, 0.3, 0.25, 0.2, 0.1]
mpeg_qps = [30]

ground_truth_dict = read_results_dict('results/drive_sing_cut10fps_gt')

with open('stats_drive_sing_cut10fps', "r") as f:
    lines = f.readlines()
    f.close()

f1_score_list = []
for res in mpeg_resolutions:
    for qp in mpeg_qps:
        video_name = f'results/drive_sing_cut10fps_mpeg_{res}_{qp}'
        rdict = read_results_dict(video_name)
        # results = merge_boxes_in_results(rdict, 0.3, 0.3)
        results = Results()
        for fid, regions in rdict.items():
            for r in regions:
                results.append(r)
        f1, stats = evaluate(results, ground_truth_dict, 0.3)
        # Write evaluation results to file
        f1_score_list.append(f1)
        print(f1)
        # if qp == 40 and res == 0.5:
        #     print('low config', f1, stats)
        # # if qp == 40 and res == 0.375:
        # #     print('low config', f1)
        # if qp == 36 and res == 0.75:
        #     print('high config', f1, stats)

for idx, line in enumerate(lines):
    if idx == 0:
        continue
    line = line.split(",")
    if idx == 1:
        # skip first line
        f1_list.append(1.)
        bw_list.append(1.)
        # res_list.append(1.)
        # qp_list.append(max(line[3],line[4]))
        notation_list.append('gt')
        continue
    bw = float(line[-3])/float(lines[1].split(",")[-3]) #gt
    f1 = f1_score_list[idx-2]
    f1_list.append(f1)
    bw_list.append(bw)
    notation_list.append(str(max(float(line[1]), float(line[2]))) \
                         + '/' + str(max(int(line[3]),int(line[4]))))

# sort by bw

ax = plt.subplot(111)

# bw =
# f1 =
# import pdb; pdb.set_trace()
bw_list_sort_index = [i[0] for i in sorted(enumerate(bw_list), key=lambda x:x[1])]
bw_list_sorted = [bw_list[idx] for idx in bw_list_sort_index]
f1_list_sorted = [f1_list[idx] for idx in bw_list_sort_index]
notation_list_sorted = [notation_list[idx] for idx in bw_list_sort_index]

# no gt
line, = plt.plot(bw_list_sorted[:-1], f1_list_sorted[:-1], 'go')

for i in range(len(notation_list_sorted[:-1])):
    plt.annotate(notation_list_sorted[i], xy=(bw_list_sorted[i], f1_list_sorted[i]),\
                                # xytext = (bw_list_sorted[i], f1_list_sorted[i] + 0.01), \
                                # arrowprops=dict(facecolor='black', shrink=0.01),
                                )

plt.ylim(0.4, 0.8)
plt.xlim(0., 1.)
plt.savefig('mpeg_curve_multiscan.png')
