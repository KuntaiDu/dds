import numpy as np
from dds_utils import *
import os

result_DIR = './backend/no_filter_combined_merged_bboxes'
output_DIR = './backend/no_filter_denoise_combined_merged_bboxes'
num_frames = 661
batch_size = 3
iou_threshold = 0.1
os.makedirs(output_DIR, exist_ok=True)
fnames = sorted([f for f in os.listdir(result_DIR) if "req_regions" not in f])
for file in fnames:
    print(file)
    rdict = read_results_dict(os.path.join(result_DIR, file))
    for iframe in range(0, num_frames, batch_size):
        start_id = iframe
        end_id = min(num_frames, iframe + batch_size)
        for i in range(start_id, end_id):
            if i in rdict:
                for i_object in rdict[i]:
                    cnt = 0
                    for j in range(start_id, end_id):
                        if i != j:
                            if j in rdict:
                                for j_object in rdict[j]:
                                    iou = calc_iou(i_object, j_object)
                                    if iou > iou_threshold:
                                        cnt += 1
                                        break
                    # print(cnt)
                    if cnt < 1:
                        rdict[i].remove(i_object)

    results = Results()
    for id, region in rdict.items():
        for single_region in region:
            results.add_single_result(single_region)

    results.fill_gaps(num_frames)
    results.write(os.path.join(output_DIR, file))
    # import pdb; pdb.set_trace()
