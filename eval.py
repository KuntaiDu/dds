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
mpeg_resolutions = [0.2, 0.375, 0.5, 0.75, 0.8, 0.9]
mpeg_qps = [30]

ground_truth_dict = read_results_dict('results/drive_sing_cut10fps_gt')

f1_score_list = []
for res in mpeg_resolutions:
    for qp in mpeg_qps:
        video_name = f'results/drive_sing_cut10fps_mpeg_{res}_{qp}'
        rdict = read_results_dict(video_name)
        results = merge_boxes_in_results(rdict, 0.3, 0.3)
        f1, stats = evaluate(results, ground_truth_dict, 0.3)
        # Write evaluation results to file
        f1_score_list.append(f1)
