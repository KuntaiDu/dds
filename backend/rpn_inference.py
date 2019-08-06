import os
import logging
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.compat.v1 import ConfigProto
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.utils import shape_utils
# from object_detection.utils import ops as utils_ops
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

import six.moves.urllib as urllib
import sys
import tarfile

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

        self.logger.info("Object detector initialized")

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
            RPN_box_tensor = tf.compat.v1.convert_to_tensor(output_dict[key_tensor_map['RPN_box_no_normalized']])
            RPN_shape_tensor = tf.compat.v1.convert_to_tensor(output_dict[key_tensor_map['Resized_shape']])
            RPN_box_tensor = tf.expand_dims(RPN_box_tensor, 0)
            RPN_shape_tensor = tf.expand_dims(RPN_shape_tensor, 0)

            def normalize_boxes(args):
                proposal_boxes_per_image = args[0]
                image_shape = args[1]
                normalized_boxes_per_image = box_list_ops.to_normalized_coordinates(
                    box_list.BoxList(proposal_boxes_per_image), image_shape[0],
                    image_shape[1], check_range=False).get()
                return normalized_boxes_per_image

            normalized_proposal_boxes = shape_utils.static_or_dynamic_map_fn(
                normalize_boxes, elems=[RPN_box_tensor, RPN_shape_tensor], \
                                        dtype=tf.float32)
            output_dict['RPN_box_normalized'] = tf.compat.v1.squeeze(normalized_proposal_boxes).eval(session = self.session)
            inds = output_dict[key_tensor_map['RPN_score']] > 0.95

            # create new keys
            output_dict['RPN_box_normalized'] = output_dict['RPN_box_normalized'][inds,:]
            output_dict['RPN_score'] = output_dict[key_tensor_map['RPN_score']][inds]

        return output_dict

    # def infer(self, image_np):
    #     output_dict = self.run_inference_for_single_image(image_np,
    #                                                       self.d_graph)
    #     # The results array will have (class, (xmin, xmax, ymin, ymax)) tuples
    #     results = []
    #     for i in range(len(output_dict['detection_boxes'])):
    #         object_class = output_dict['detection_classes'][i]
    #         relevant_class = False
    #         for k in Detector.classes.keys():
    #             if object_class in Detector.classes[k]:
    #                 object_class = k
    #                 relevant_class = True
    #                 break
    #         if not relevant_class:
    #             continue
    #
    #         ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
    #         confidence = output_dict['detection_scores'][i]
    #         box_tuple = (xmin, ymin, xmax - xmin, ymax - ymin)
    #         results.append((object_class, confidence, box_tuple))
    #     return results

def main():
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
    # prepare data
    PATH_TO_TEST_IMAGES_DIR = '/data/yuanx/origin'
    PATH_TO_SAVE_IMAGES_DIR = '/data/yuanx/tf_rpn_origin'
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{:010d}.png'.format(i)) for i in range(0, 320) ]
    SAVE_IMAGE_PATHS = [ os.path.join(PATH_TO_SAVE_IMAGES_DIR, '{:010d}.jpg'.format(i)) for i in range(0, 320) ]

    # Size, in inches, of the output images.
    for idx, image_path in enumerate(TEST_IMAGE_PATHS):
        print(idx)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Actual detection.
        output_dict = detector.run_inference_rpn_for_single_image(image, detector.d_graph)
        for region in output_dict['RPN_box_normalized']:
            # y_min, x_min, y_max, x_max
            # bbox = [int(np.round(x)) for x in region]
            x_min = int(np.round(region[1] * image.shape[1]))
            y_min = int(np.round(region[0] * image.shape[0]))
            x_max = int(np.round(region[3] * image.shape[1]))
            y_max = int(np.round(region[2] * image.shape[0]))
            image = cv2.rectangle(image, (x_min, y_min), \
                                  (x_max, y_max), \
                                  (0, 204, 0), 2)

        cv2.imwrite(SAVE_IMAGE_PATHS[idx], image)

if __name__== "__main__":
    main()
