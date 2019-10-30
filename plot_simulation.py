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


data_dir = "/data/kuntai/new_dataset"
# mpeg_resolutions = [0.15, 0.2, 0.25, 0.375, 0.5, 0.625, 0.75]
mpeg_resolutions = [0.2, 0.375, 0.5, 0.75, 0.8, 0.9]
mpeg_qps = [30, 32, 36, 38, 40]

ground_truth_dict = read_results_dict('results/out10fps720p_mpeg_0.9_30')

with open('stats_new_gt', "r") as f:
    lines = f.readlines()
    f.close()

f1_score_list = []
for res in mpeg_resolutions:
    for qp in mpeg_qps:
        video_name = f'results/out10fps720p_mpeg_{res}_{qp}'
        rdict = read_results_dict(video_name)
        results = merge_boxes_in_results(rdict, 0.3, 0.3)
        # results = Results()
        # for fid, regions in rdict.items():
        #     for r in regions:
        #         results.append(r)
        f1, stats = evaluate(results, ground_truth_dict, 0.3)
        # Write evaluation results to file
        f1_score_list.append(f1)
        if qp == 40 and res == 0.5:
            print('low config', f1, stats)
        # if qp == 40 and res == 0.375:
        #     print('low config', f1)
        if qp == 36 and res == 0.75:
            print('high config', f1, stats)

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
        notation_list.append('gt:' + str(1.0) + '/' + str(max(line[3],line[4])))
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

ground_truth_dict = read_results_dict('results/out10fps720p_mpeg_0.9_30')
# video_name = 'results/highway_0_00_00_00_mpeg_0.375_40'
'''
video_name = 'results/out10fps720p_dds_0.5_0.75_40_36'
rdict = read_results_dict(video_name)
# for i in range(11):
#     for j in range(11):
#         for k in range(11):
# results = merge_boxes_in_results(rdict, 0.3, 0.3)
results = Results()
for fid, regions in rdict.items():
    for r in regions:
        results.append(r)
f1_dds, _ = evaluate(results, ground_truth_dict, 0.3)
print('f1_dds', f1_dds)
plt.plot( (745216)/4735062.0, f1_dds, 'ro')
plt.annotate('0.5/0.75',xy=((745216)/4735062.0, f1_dds))

video_name = 'results/out10fps720p_dds_0.375_0.75_40_36'
rdict = read_results_dict(video_name)
# for i in range(11):
#     for j in range(11):
#         for k in range(11):
# results = merge_boxes_in_results(rdict, 0.3, 0.3)
results = Results()
for fid, regions in rdict.items():
    for r in regions:
        results.append(r)
f1_dds, _ = evaluate(results, ground_truth_dict, 0.3)
print('f1_dds', f1_dds)
plt.plot( (529037)/4735062.0, f1_dds, 'ro')
plt.annotate('0.375/0.75',xy=((529037)/4735062.0, f1_dds))


video_name = 'results/out10fps720p_dds_0.375_0.9_40_32'
rdict = read_results_dict(video_name)
# for i in range(11):
#     for j in range(11):
#         for k in range(11):
# results = merge_boxes_in_results(rdict, 0.3, 0.3)
results = Results()
for fid, regions in rdict.items():
    for r in regions:
        results.append(r)
f1_dds, _ = evaluate(results, ground_truth_dict, 0.3)
print('f1_dds', f1_dds)
plt.plot( (589096)/4735062.0, f1_dds, 'ro')
plt.annotate('0.375/0.9',xy=((589096)/4735062.0, f1_dds))


video_name = 'results/out10fps720p_dds_0.5_0.9_40_32'
rdict = read_results_dict(video_name)
# for i in range(11):
#     for j in range(11):
#         for k in range(11):
# results = merge_boxes_in_results(rdict, 0.3, 0.3)
results = Results()
for fid, regions in rdict.items():
    for r in regions:
        results.append(r)
f1_dds, _ = evaluate(results, ground_truth_dict, 0.3)
print('f1_dds', f1_dds)
plt.plot( (789335)/4735062.0, f1_dds, 'ro')
plt.annotate('0.5/0.9',xy=((789335)/4735062.0, f1_dds))
'''


# video_name = 'results_back_02/out10fps720p_dds_0.5_0.8_40_32'
# rdict = read_results_dict(video_name)
# # for i in range(11):
# #     for j in range(11):
# #         for k in range(11):
# # results = merge_boxes_in_results(rdict, 0.3, 0.3)
# results = Results()
# for fid, regions in rdict.items():
#     for r in regions:
#         results.append(r)
# f1_dds, _ = evaluate(results, ground_truth_dict, 0.3)
# print('f1_dds', 0.6439716312056738)
# plt.plot( (776948)/4735062.0, 0.6439716312056738, 'ro')
# plt.annotate('0.5/0.8',xy=((776948)/4735062.0, f1_dds))

video_name = 'results_back_0811/out10fps720p_dds_0.5_0.75_40_36'
rdict = read_results_dict(video_name)
# for i in range(11):
#     for j in range(11):
#         for k in range(11):
results = merge_boxes_in_results(rdict, 0.3, 0.3)
# results = Results()
# for fid, regions in rdict.items():
#     for r in regions:
#         results.append(r)
f1_dds, stats = evaluate(results, ground_truth_dict, 0.3)
print('f1_dds',f1_dds, stats)
plt.plot( (1112757)/4735062.0, f1_dds, 'ro')
plt.annotate('0.5/0.8',xy=((1112757)/4735062.0, f1_dds))
''



# results/out10fps720p_dds_0.5_0.75_40_36,0.5,0.75,40,36,661,0.3,0.3,4,347.0,225.0,406.0,0.5237735849056604,681033,232712,913745,661,emulation
# plt.plot( 399489/1996197.0, 0.5695862439548629, 'ro')
# results/highway_0_00_00_00_dds_0.375_0.75_40_36,0.375,0.75,40,36,15,0.3,0.3,4,2120.0,1053.0,2151.0,0.5695862439548629,236863,386195,623058,320,emulation
# results/highway_0_00_00_00_dds_0.375_0.75_40_36,0.375,0.75,40,36,15,0.3,0.3,4,2006.0,1002.0,2265.0,0.5511746118972387,236863,368497,605360,320,emulation
# results/highway_0_00_00_00_dds_0.375_0.75_40_36,0.375,0.75,40,36,15,0.3,0.3,4,1908.0,1035.0,2363.0,0.5289714444136402,236863,354616,591479,320,emulation

plt.ylim(0., 1.)
# plt.xlim(0., 0.5)
plt.savefig('mpeg_curve_gt_temporal_smooth.png')
