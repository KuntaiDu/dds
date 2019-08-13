import os
import numpy as np
import cv2
import sys
sys.path.append("../")
from merger import *
from dds_utils import *

PATH_TO_SAVE_IMAGES_DIR = '/data/yuanx/out10fps720p_compare_STATS'
os.makedirs(PATH_TO_SAVE_IMAGES_DIR, exist_ok=True)
result_list=[
    '/home/yuanx/dds-simulation/results/drive_sing_cut10fps_mpeg_0.375_30', #rpn+rcnn low mpeg
    '/home/yuanx/dds-simulation/results/drive_sing_cut10fps_gt', # 2nd iter
]
# BGR
# RPN slect, RCNN high, DDS final
color_map = [(0, 0 , 255), (0, 255,0 )]

VIDEO_NAME = 'drive_sing_cut10fps'
# PATH_TO_SAVE_IMAGES_DIR = '/data/yuanx/out10fps720p'
PATH_TO_ORIGIN_IMAGES_DIR = '/data/yuanx/new_dataset/drive_sing_cut10fps/src/'


ORIGIN_IMAGE_PATH = [os.path.join(PATH_TO_ORIGIN_IMAGES_DIR, '{:010d}.png'.format(i)) for i in range(0, 661)]
SAVE_IMAGE_PATHS = [ os.path.join(PATH_TO_SAVE_IMAGES_DIR, '{:010d}.jpg'.format(i)) for i in range(0, 661) ]

result_dict = []
for i in result_list:
    result_dict.append(read_results_txt_dict(i))

# Size, in inches, of the output images.
final_results = Results()
RCNN_results = Results()
for idx, image_path in enumerate(ORIGIN_IMAGE_PATH):
    print(image_path)
    image = cv2.imread(image_path)
    for idx_result_dict, single_result_dict in enumerate(result_dict):
        if idx in single_result_dict:
            for res in single_result_dict[idx]:
                if res.label != 'vehicle' and res.label != 'object':
                    continue
                x = int(np.round(res.x * image.shape[1]))
                y = int(np.round(res.y * image.shape[0]))
                w = int(np.round(res.w * image.shape[1]))
                h = int(np.round(res.h * image.shape[0]))
                image = cv2.rectangle(image, (x,y), (x+w, y+h), color_map[idx_result_dict] , 1)

        # single_result = merge_boxes_in_results(single_result_dict, 0.3, 0.3)
        # for res in single_result.regions:
        #     # import pdb; pdb.set_trace()
        #     if ( res.label != 'vehicle' and res.label != 'object') or res.conf < 0.3  or res.w * res.h > 0.01:
        #         continue
        #     if res.fid == idx:
        #         x = int(np.round(res.x * image.shape[1]))
        #         y = int(np.round(res.y * image.shape[0]))
        #         w = int(np.round(res.w * image.shape[1]))
        #         h = int(np.round(res.h * image.shape[0]))
        #         image = cv2.rectangle(image, (x,y), (x+w, y+h), color_map[idx_result_dict] , 1)
    cv2.imwrite(SAVE_IMAGE_PATHS[idx], image)
