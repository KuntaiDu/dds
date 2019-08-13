import numpy as np
import sys
import os
sys.path.append("../")
from merger import *
from dds_utils import *

IMAGE_DIR = '/data/yuanx/new_dataset/out10fps720p/src/'
ORIGIN_IMAGE_PATH = [os.path.join(IMAGE_DIR, '{:010d}.png'.format(i)) for i in range(0, 661)]

req_regions_path = '../results_back_0811/out10fps720p_dds_0.5_0.75_40_36_req_regions_0.3_0.3'
results_dict = read_results_txt_dict(req_regions_path)
all_req_regions = Results()

for fid in range(len(ORIGIN_IMAGE_PATH)):
    if fid not in results_dict:
        continue
    fid_results = results_dict[fid]
    for single_result in fid_results:
        all_req_regions.add_single_result(single_result)

regions_size = compute_area_of_regions(all_req_regions)
print('regions_size', regions_size)

origin_total_pixels_normalized = 0.
for idx, image_path in enumerate(ORIGIN_IMAGE_PATH):
    image = cv2.imread(image_path)
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = frame == 0
    all_black = mask.sum(axis=2) == 3
    origin_single_pixels_normalized = 1.0 - np.sum(all_black)/(image.shape[0] * image.shape[1])
    origin_total_pixels_normalized += origin_single_pixels_normalized

print('origin_total_pixels_normalized', origin_total_pixels_normalized)
print('pixel ratio', regions_size/origin_total_pixels_normalized)
