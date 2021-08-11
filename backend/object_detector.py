import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
import torchvision.transforms as T
import torch
# from object_detection.utils import ops as utils_ops
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'


import yaml
with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())
relevant_classes = dds_env['relevant_classes']
print(relevant_classes)

class Detector:
    classes = {
        "vehicle": [3, 6, 7, 8],
        "persons": [1, 2, 4],
        "roadside-objects": [10, 11, 13, 14]
    }

    def __init__(self, model_path='frozen_inference_graph.pb'):

        # dirty fix
        # MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
        # MODEL_FILE = MODEL_NAME + '.tar.gz'
        # DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
        # SAVE_BASE = '/data/kuntai/model/'
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        model_path = dds_env['resnet_101_coco']

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


    def split_image(self, image):
        # split image to a batch of 1024*576 image
        if image.shape[1] < 0.55 * 1920:
        # if image.shape[1] < 1921:
            return image, None
        cropped_images = []
        offsets = []
        step_x = 1024 # to make sure only 4 areas happens
        step_y = 576 # to make sure only 4 areas happens
        # cropped_image = np.zeros((1024,576,3), np.uint8)
        for move_time_x in range(int((image.shape[1]-1024)/step_x) + 2): #1024
            for move_time_y in range(int((image.shape[0]-576)/step_y) + 2): #576
                x_max = move_time_x * step_x + 1024
                if x_max > image.shape[1]:
                    x_min = image.shape[1] - 1024
                    x_max = image.shape[1]
                else:
                    x_min = move_time_x * step_x

                y_max = move_time_y * step_y + 576
                if y_max > image.shape[0]:
                    y_min = image.shape[0] - 576
                    y_max = image.shape[0]
                else:
                    y_min = move_time_y * step_y
                cropped_image = image[y_min:y_max, x_min:x_max, :]
                cropped_images.append(cropped_image)
                offsets.append((x_min,y_min))

        return cropped_images, offsets

    def split_image_no_overlapping(self, image):
        # split image into 4 blocks
        cropped_images = []
        offsets = []
        step_x = int(image.shape[1]/4.) # to make sure only 4 areas happens
        step_y = int(image.shape[0]/4.)
        # cropped_image = np.zeros((1024,576,3), np.uint8)
        for move_time_x in range(4): #38
            for move_time_y in range(4): #576
                x_min = int(move_time_x * step_x)
                x_max = int(min((move_time_x + 1) * step_x, image.shape[1]))
                y_min = int(move_time_y * step_y)
                y_max = int(min((move_time_y + 1) * step_y, image.shape[0]))

                cropped_image = image[y_min:y_max, x_min:x_max, :]
                cropped_images.append(cropped_image)
                offsets.append((x_min,y_min,x_max,y_max))

        return cropped_images, offsets

    def collect_results_from_all_images_no_overlapping(self, original_image, output_dict_list, offsets):
        new_output_dict = {}
        new_output_dict['num_detections'] = 0
        new_output_dict['detection_boxes'] = []
        new_output_dict['detection_classes'] = []
        new_output_dict['detection_scores'] = []
        multi_scan_results = [[]]*16
        for i in range(len(output_dict_list)):
            new_output_dict['num_detections'] += output_dict_list[i]['num_detections']
            # ymin, xmin, ymax, xmax
            x_min, y_min, x_max, y_max = offsets[i]
            single_scan_results = []
            for j in range(len(output_dict_list[i]['detection_boxes'])):
                # if RECORD:
                object_class = output_dict_list[i]['detection_classes'][j]
                relevant_class = False
                for k in Detector.classes.keys():
                    if object_class in Detector.classes[k]:
                        object_class = k
                        relevant_class = True
                        break
                if not relevant_class:
                    continue

                ymin, xmin, ymax, xmax = output_dict_list[i]['detection_boxes'][j]
                confidence = output_dict_list[i]['detection_boxes'][j]
                box_tuple = (xmin, ymin, xmax - xmin, ymax - ymin)
                single_scan_results.append((object_class, confidence, box_tuple))

                output_dict_list[i]['detection_boxes'][j][0] = (output_dict_list[i]['detection_boxes'][j][0]*(y_max - y_min + 1) + y_min)/original_image.shape[0]
                output_dict_list[i]['detection_boxes'][j][1] = (output_dict_list[i]['detection_boxes'][j][1]*(x_max - x_min + 1)  + x_min)/original_image.shape[1]
                output_dict_list[i]['detection_boxes'][j][2] = (output_dict_list[i]['detection_boxes'][j][2]*(y_max - y_min + 1) + y_min)/original_image.shape[0]
                output_dict_list[i]['detection_boxes'][j][3] = (output_dict_list[i]['detection_boxes'][j][3]*(x_max - x_min + 1) + x_min)/original_image.shape[1]

            new_output_dict['detection_boxes'].extend(output_dict_list[i]['detection_boxes'])
            new_output_dict['detection_classes'].extend(output_dict_list[i]['detection_classes'])
            new_output_dict['detection_scores'].extend(output_dict_list[i]['detection_scores'])

            multi_scan_results[i] = single_scan_results


        new_output_dict['detection_boxes'] = np.stack(new_output_dict['detection_boxes'], axis=0)
        new_output_dict['detection_classes'] = np.stack(new_output_dict['detection_classes'], axis=0)
        new_output_dict['detection_scores'] = np.stack(new_output_dict['detection_scores'], axis=0)
        return new_output_dict, multi_scan_results

    def collect_results_from_all_images(self, original_image, output_dict_list, offsets):
        new_output_dict = {}
        new_output_dict['num_detections'] = 0
        new_output_dict['detection_boxes'] = []
        new_output_dict['detection_classes'] = []
        new_output_dict['detection_scores'] = []
        multi_scan_results = []
        for i in range(len(output_dict_list)):
            new_output_dict['num_detections'] += output_dict_list[i]['num_detections']
            # ymin, xmin, ymax, xmax
            x_min, y_min = offsets[i]
            single_scan_results = []
            for j in range(len(output_dict_list[i]['detection_boxes'])):
                # if RECORD:
                object_class = output_dict_list[i]['detection_classes'][j]
                relevant_class = False
                for k in Detector.classes.keys():
                    if object_class in Detector.classes[k]:
                        object_class = k
                        relevant_class = True
                        break
                if not relevant_class:
                    continue

                ymin, xmin, ymax, xmax = output_dict_list[i]['detection_boxes'][j]
                confidence = output_dict_list[i]['detection_boxes'][j]
                box_tuple = (xmin, ymin, xmax - xmin, ymax - ymin)
                single_scan_results.append((object_class, confidence, box_tuple))

                output_dict_list[i]['detection_boxes'][j][0] = (output_dict_list[i]['detection_boxes'][j][0]*576 + y_min)/original_image.shape[0]
                output_dict_list[i]['detection_boxes'][j][1] = (output_dict_list[i]['detection_boxes'][j][1]*1024 + x_min)/original_image.shape[1]
                output_dict_list[i]['detection_boxes'][j][2] = (output_dict_list[i]['detection_boxes'][j][2]*576 + y_min)/original_image.shape[0]
                output_dict_list[i]['detection_boxes'][j][3] = (output_dict_list[i]['detection_boxes'][j][3]*1024 + x_min)/original_image.shape[1]

            new_output_dict['detection_boxes'].extend(output_dict_list[i]['detection_boxes'])
            new_output_dict['detection_classes'].extend(output_dict_list[i]['detection_classes'])
            new_output_dict['detection_scores'].extend(output_dict_list[i]['detection_scores'])

            multi_scan_results.append(single_scan_results)


        new_output_dict['detection_boxes'] = np.stack(new_output_dict['detection_boxes'], axis=0)
        new_output_dict['detection_classes'] = np.stack(new_output_dict['detection_classes'], axis=0)
        new_output_dict['detection_scores'] = np.stack(new_output_dict['detection_scores'], axis=0)
        return new_output_dict, multi_scan_results

    def run_inference_for_single_image(self, image, graph):
        with self.d_graph.as_default():
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops
                                for output in op.outputs}
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes',
                    'detection_scores', 'detection_classes',
                    'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = (tf.compat.v1.get_default_graph()
                                        .get_tensor_by_name(tensor_name))

            image_tensor = (tf.compat.v1.get_default_graph()
                            .get_tensor_by_name('image_tensor:0'))
            # Run inference
            feed_dict = {image_tensor: np.expand_dims(image, 0)}
            output_dict = self.session.run(tensor_dict,
                                           feed_dict=feed_dict)

            # all outputs are float32 numpy arrays,
            # so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = (
                output_dict['detection_boxes'][0])
            output_dict['detection_scores'] = (
                output_dict['detection_scores'][0])
        return output_dict

    def infer(self, image_np):
        # imgae_crops, offsets = self.split_image_no_overlapping(image_np)

        # secretly super-resolute image here
        image_crops = image_np
        offsets = None
        multi_scan_results = None
        if not offsets:
            output_dict = self.run_inference_for_single_image(image_crops,
                                                            self.d_graph)
        else:
            output_dict_list = []
            for i in range(len(offsets)):
                output_dict_list.append(self.run_inference_for_single_image(image_crops[i],
                                                                self.d_graph))

            output_dict, multi_scan_results = self.collect_results_from_all_images_no_overlapping(image_np, output_dict_list, offsets)


        # The results array will have (class, (xmin, xmax, ymin, ymax)) tuples
        results = []
        for i in range(len(output_dict['detection_boxes'])):
            object_class = output_dict['detection_classes'][i]
            relevant_class = False
            for k in Detector.classes.keys():
                if object_class in Detector.classes[k]:
                    object_class = k
                    relevant_class = True
                    break
            if not relevant_class:
                continue

            ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
            confidence = output_dict['detection_scores'][i]
            box_tuple = (xmin, ymin, xmax - xmin, ymax - ymin)
            results.append((object_class, confidence, box_tuple))

        return results, image_crops, offsets
