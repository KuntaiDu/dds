import os
import logging
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.compat.v1 import ConfigProto
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.utils import shape_utils
import glob
import sys
sys.path.append("../")
from merger import *
from dds_utils import read_results_txt_dict
# from object_detection.utils import ops as utils_ops
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

import six.moves.urllib as urllib
import sys
import tarfile
PROJ_ROOT_DIR = '/home/yuanx/dds-simulation/'
class Detector:
    classes = {
        "vehicle": [3, 6, 7, 8],
        "persons": [1, 2, 4],
        "roadside-objects": [10, 11, 13, 14]
    }

    def __init__(self, model_path='frozen_inference_graph.pb'):
        self.logger = logging.getLogger("object_detector")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        self.model_path = model_path
        self.d_graph = tf.Graph()
        with self.d_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.session = tf.compat.v1.Session(config=config)


    def run_inference_rpn_for_single_image(self, image, graph):
        with self.d_graph.as_default():
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops
                                for output in op.outputs}
            tensor_dict = {}
            key_tensor_map = {
                'RPN_box_no_normalized' : 'BatchMultiClassNonMaxSuppression/map/while/MultiClassNonMaxSuppression/Gather/Gather:0',
                'RPN_score' : 'BatchMultiClassNonMaxSuppression/map/while/MultiClassNonMaxSuppression/Gather/Gather_2:0',
                'Resized_shape' : 'Preprocessor/map/while/ResizeToRange/stack_1:0',
            }
            for key, tensor_name in key_tensor_map.items():
                if tensor_name in all_tensor_names:
                    tensor_dict[tensor_name] = (tf.compat.v1.get_default_graph()
                                        .get_tensor_by_name(tensor_name))

            image_tensor = (tf.compat.v1.get_default_graph()
                            .get_tensor_by_name('image_tensor:0'))
            # Run inference
            feed_dict = {image_tensor: np.expand_dims(image, 0)}
            output_dict = self.session.run(tensor_dict,
                                           feed_dict=feed_dict)
            # nromalize the width and height
            # height, width, channel = Resized_shape_2
            # filter the output_dict
            # y_min, x_min, y_max, x_max
            # normalize proposal boxes
            # RPN_box_tensor = tf.compat.v1.convert_to_tensor(output_dict[key_tensor_map['RPN_box_no_normalized']])
            # RPN_shape_tensor = tf.compat.v1.convert_to_tensor(output_dict[key_tensor_map['Resized_shape']])
            # RPN_box_tensor = tf.expand_dims(RPN_box_tensor, 0)
            # RPN_shape_tensor = tf.expand_dims(RPN_shape_tensor, 0)

            # THIS is TOO SLOW!!!
            # def normalize_boxes(args):
            #     proposal_boxes_per_image = args[0]
            #     image_shape = args[1]
            #     normalized_boxes_per_image = box_list_ops.to_normalized_coordinates(
            #         box_list.BoxList(proposal_boxes_per_image), image_shape[0],
            #         image_shape[1], check_range=False).get()
            #     return normalized_boxes_per_image
            #
            # normalized_proposal_boxes = shape_utils.static_or_dynamic_map_fn(
            #     normalize_boxes, elems=[RPN_box_tensor, RPN_shape_tensor], \
            #                             dtype=tf.float32)

            # output_dict['RPN_box_normalized'] = tf.compat.v1.squeeze(normalized_proposal_boxes).eval(session = self.session)
            w = output_dict[key_tensor_map['Resized_shape']][1]
            h = output_dict[key_tensor_map['Resized_shape']][0]
            input_shape_array = np.array([h, w, h, w])
            # import pdb; pdb.set_trace()
            output_dict['RPN_box_normalized'] = output_dict[key_tensor_map['RPN_box_no_normalized']]/input_shape_array[np.newaxis,:]
            output_dict['RPN_score'] = output_dict[key_tensor_map['RPN_score']]
            # inds = output_dict[key_tensor_map['RPN_score']] > 0.9

            # create new keys
            # output_dict['RPN_box_normalized'] = output_dict['RPN_box_normalized'][inds,:]


        return output_dict

def run_rpn_inference(video, threshold_RPN, threshold_RCNN, threshold_GT, low_scale, low_qp, high_scale, high_qp, results_dir):
    # download models
    MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    SAVE_BASE = '/data/yuanx/'
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = SAVE_BASE + MODEL_NAME + '/frozen_inference_graph.pb'

    # download the model
    # if not os.path.isfile(SAVE_BASE + MODEL_FILE):
    #     opener = urllib.request.URLopener()
    #     opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, SAVE_BASE + MODEL_FILE)
    # tar_file = tarfile.open(SAVE_BASE + MODEL_FILE)
    # for file in tar_file.getmembers():
    #     file_name = os.path.basename(file.name)
    #     if 'frozen_inference_graph.pb' in file_name:
    #         tar_file.extract(file, SAVE_BASE)

    # List of the strings that is used to add correct label for each box.
    # PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    # initial detector
    detector = Detector(PATH_TO_FROZEN_GRAPH)

    VIDEO_NAME = video
    threshold_RPN = threshold_RPN
    threshold_RCNN = 0.3
    threshold_GT = 0.3
    scale = low_scale
    qp = low_qp
    high_scale = high_scale
    high_qp = high_qp
    RESULT_DIR = results_dir

    # print("PROJ_ROOT_DIR:", PROJ_ROOT_DIR)

    PATH_TO_HIGH_FILE = f'{RESULT_DIR}/{VIDEO_NAME}_mpeg_{high_scale}_{high_qp}'
    PATH_TO_FINAL = f'{RESULT_DIR}/{VIDEO_NAME}_mpeg_{scale}_{qp}'

    PATH_TO_TEST_IMAGES_DIR = f'/data/yuanx/new_dataset/{VIDEO_NAME}_{scale}_{qp}/src/' #0.375, 40

    PATH_TO_SAVE_IMAGES_DIR = f'/data/yuanx/visualization/{VIDEO_NAME}_RPN_{threshold_RPN}'
    os.makedirs(PATH_TO_SAVE_IMAGES_DIR, exist_ok=True)

    PATH_TO_ORIGIN_IMAGES_DIR = f'/data/yuanx/new_dataset/{VIDEO_NAME}/src/'
    number_of_frames = len(glob.glob1(PATH_TO_ORIGIN_IMAGES_DIR,"*.png"))
    print(number_of_frames)


    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{:010d}.png'.format(i)) for i in range(0, number_of_frames) ]
    ORIGIN_IMAGE_PATH = [os.path.join(PATH_TO_ORIGIN_IMAGES_DIR, '{:010d}.png'.format(i)) for i in range(0, number_of_frames)]
    SAVE_IMAGE_PATHS = [ os.path.join(PATH_TO_SAVE_IMAGES_DIR, '{:010d}.jpg'.format(i)) for i in range(0, number_of_frames) ]
    VIS = False

    result_dict_gt = read_results_txt_dict(PATH_TO_HIGH_FILE)
    result_dict_rcnn = read_results_txt_dict(PATH_TO_FINAL)
    # dirty fix

    # Size, in inches, of the output images.
    final_results = Results()
    RCNN_results = Results()
    for idx, image_path in enumerate(TEST_IMAGE_PATHS):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # rpn detection, return a dict
        result_rpn = detector.run_inference_rpn_for_single_image(image, detector.d_graph)
        # read_gt

        # if VIS:
        # draw gt, threshold=0.3
        # for res in result_dict_gt[idx]:
        #     x = int(np.round(res.x * image_origin.shape[1]))
        #     y = int(np.round(res.y * image_origin.shape[0]))
        #     w = int(np.round(res.w * image_origin.shape[1]))
        #     h = int(np.round(res.h * image_origin.shape[0]))
        #     score = res.conf
        #     # if score > 0.3 and res.w*res.h <0.04:
        #     # image_origin = cv2.rectangle(image_origin, (x,y), (x+w, y+h), (0, 255, 0), 1)
        #         # cv2.putText(im, '%.3f' % (score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
        #         #             1.0, (0, 0, 255), thickness=1)
        # for res in result_dict_rcnn[idx]:
        #     x = int(np.round(res.x * image_origin.shape[1]))
        #     y = int(np.round(res.y * image_origin.shape[0]))
        #     w = int(np.round(res.w * image_origin.shape[1]))
        #     h = int(np.round(res.h * image_origin.shape[0]))
        #     score = res.conf
            # if score > 0.3 and res.w*res.h < 0.04:
            # image_origin = cv2.rectangle(image_origin, (x,y), (x+w, y+h), (255, 0 , 0), 1)
                # cv2.putText(im, '%.3f' % (score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                #             1.0, (0, 0, 255), thickness=1)

        # for res in results_rpn:
        #     res[]
        # y_min, x_min, y_max, x_max
        RPN_Regions = []
        frame_with_no_results = True
        for idx_region, region in enumerate(result_rpn['RPN_box_normalized']):
            x = region[1]
            y = region[0]
            w = region[3] - region[1]
            h = region[2] - region[0]
            conf = result_rpn['RPN_score'][idx_region]
            if conf < threshold_RPN or w*h == 0. or w*h > 0.04:
                continue
            # print((idx, idx_region))
            single_region = Region(idx, x, y, w, h, conf, 'object',
                                   scale, 'generic')
            RPN_Regions.append(single_region)
            frame_with_no_results = False

        if frame_with_no_results:
            RPN_Regions.append(
                Region(idx, 0, 0, 0, 0, 0.1, "no obj", scale))
        '''
        overlap_pairwise_list = pairwise_overlap_indexing_list(
            RPN_Regions, 0.3)
        overlap_graph = to_graph(overlap_pairwise_list)
        grouped_bbox_idx = [c for c in sorted(
            connected_components(overlap_graph), key=len, reverse=True)]
        merged_RPN_regions = simple_merge(RPN_Regions, grouped_bbox_idx)
        '''
        # print(len(RPN_Regions))
        # if VIS:
        image_origin = cv2.imread(ORIGIN_IMAGE_PATH[idx])
        draw_this_frame=False
        for res in RPN_Regions:
            if res.conf < threshold_RPN or res.w * res.h == 0. or res.w*res.h > 0.1:
                continue
            x = int(np.round(res.x * image_origin.shape[1]))
            y = int(np.round(res.y * image_origin.shape[0]))
            w = int(np.round(res.w * image_origin.shape[1]))
            h = int(np.round(res.h * image_origin.shape[0]))
            score = res.conf
            image_origin = cv2.rectangle(image_origin, (x,y), (x+w, y+h), (0, 0 , 255), 2)
            draw_this_frame = True
        if idx in result_dict_rcnn:
            for res in result_dict_rcnn[idx]:
                if res.conf < threshold_RCNN or res.w*res.h == 0. or res.w*res.h >0.04 or res.label != 'vehicle':
                    continue
                # import pdb; pdb.set_trace()
                x = int(np.round(res.x * image_origin.shape[1]))
                y = int(np.round(res.y * image_origin.shape[0]))
                w = int(np.round(res.w * image_origin.shape[1]))
                h = int(np.round(res.h * image_origin.shape[0]))

                image_origin = cv2.rectangle(image_origin, (x,y), (x+w, y+h), (255, 0 , 0), 2)
                draw_this_frame = True
        if idx in result_dict_gt:
            for res in result_dict_gt[idx]:
                if res.conf < threshold_GT or res.w*res.h == 0. or res.w*res.h >0.04 or res.label != 'vehicle':
                    continue
                # import pdb; pdb.set_trace()
                x = int(np.round(res.x * image_origin.shape[1]))
                y = int(np.round(res.y * image_origin.shape[0]))
                w = int(np.round(res.w * image_origin.shape[1]))
                h = int(np.round(res.h * image_origin.shape[0]))

                image_origin = cv2.rectangle(image_origin, (x,y), (x+w, y+h), (0, 255 , 0), 2)
                draw_this_frame = True
        if draw_this_frame:
            cv2.imwrite(SAVE_IMAGE_PATHS[idx], image_origin)

        # combine results_RPN and results_FastRCNN
        # do box merging & save
        # Fill gaps in results
        # for r in merged_RPN_regions:
        #     final_results.append(r)
        # if not VIS:
        for r in RPN_Regions:
            final_results.append(r)
        # merge again
        # import pdb; pdb.set_trace()
    for fid, rcnn_region in result_dict_rcnn.items():
        for r in rcnn_region:
            # if r.conf < 0.3 or r.w*r.h > 0.04 or r.w*r.h == 0.:
            if r.conf < threshold_RCNN or r.w*r.h == 0. or r.w*r.h >0.04 or r.label != 'vehicle':
                continue
            RCNN_results.append(r)
    final_results.combine_results(RCNN_results, 0.5)
    # write
    final_results.fill_gaps(len(TEST_IMAGE_PATHS))
    final_results.write(os.path.join(f"{PROJ_ROOT_DIR}/backend/no_filter_combined_bboxes", f'{VIDEO_NAME}_mpeg_{scale}_{qp}'))
    # read
    rdict = read_results_txt_dict(os.path.join(f"{PROJ_ROOT_DIR}/backend/no_filter_combined_bboxes", f'{VIDEO_NAME}_mpeg_{scale}_{qp}'))
    results = merge_boxes_in_results(rdict, 0.3, 0.3)
    results.fill_gaps(len(TEST_IMAGE_PATHS))
    results.write(os.path.join(f"{PROJ_ROOT_DIR}/backend/no_filter_combined_merged_bboxes", f'{VIDEO_NAME}_mpeg_{scale}_{qp}'))
    print("RPN Done")
