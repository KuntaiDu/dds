import logging
import numpy as np
import tensorflow as tf
from utils import label_map_util as lmp
from object_detection.utils import ops as utils_ops


class Detector:
    def __init__(self, model_path='frozen_inference_graph.pb',
                 labels_path='mscoco_label_map.pbtxt'):
        self.logger = logging.getLogger("object_detector")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.model_path = model_path
        self.labels_path = labels_path
        self.d_graph = tf.Graph()
        with self.d_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.session = tf.Session()
        self.category_index = lmp.create_category_index_from_labelmap(
            self.labels_path, use_display_name=True)

        self.logger.info("Object detector initialized")

    def run_inference_for_single_image(self, image, graph):
        with self.d_graph.as_default():
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
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
                    tensor_dict[key] = (tf.get_default_graph()
                                        .get_tensor_by_name(tensor_name))
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box
                # coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0],
                                           [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0],
                                           [real_num_detection, -1, -1])
                detection_masks_reframed = (
                    utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes,
                        image.shape[0], image.shape[1]))
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = (tf.get_default_graph()
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
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = (
                    output_dict['detection_masks'][0])
        return output_dict

    def infer(self, image_np):
        output_dict = self.run_inference_for_single_image(image_np,
                                                          self.d_graph)
        # The results array will have (class, (xmin, xmax, ymin, ymax)) tuples
        results = []
        for i in range(len(output_dict['detection_boxes'])):
            obj_class_index = self.category_index[
                output_dict['detection_classes'][i]]
            object_class = obj_class_index['name']
            if object_class not in ["car", "bus",
                                    "train", "truck"]:
                continue
            ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
            confidence = output_dict['detection_scores'][i]
            box_tuple = (xmin, ymin, xmax - xmin, ymax - ymin)
            results.append((object_class, confidence, box_tuple))
        return (image_np, results)
