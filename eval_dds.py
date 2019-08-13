import os
import re
import logging
import argparse
from dds_utils import *
from merger import *
# Evaluation and writing results
# Read Groundtruth results
low, high = (0,0)
f1 = 0
stats = (0, 0, 0)
ground_truth_dict = read_results_dict('results/out10fps720p_mpeg_0.9_30')
# video_name = 'results/highway_0_00_00_00_mpeg_0.375_40'
video_name = 'results/out10fps720p_dds_0.5_0.75_40_36'
rdict = read_results_dict(video_name)
# for i in range(11):
#     for j in range(11):
#         for k in range(11):
results = merge_boxes_in_results(rdict, 0.3, 0.3)
print(len(results))
# results = Results()
# for fid, regions in rdict.items():
#     for r in regions:
#         results.append(r)
f1, stats = evaluate(results, ground_truth_dict, 0.3)
print(f1)
# Write evaluation results to file
