import cv2 as cv
import numpy as np

class LocalObjectDetector:
    """Local object detector

    The object detector will use MobileNet SSD which
    is a fast, less accurate neural network"""

    def __init__(self, model_path="../MobileNetSSD.pb",
                 labels_path="../ssd_graph.pbtxt",
                 confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.net = cv.dnn.readNetFromTensorflow(model_path, labels_path)
        self.class_names = ["background", "aeroplane", "bicycle", "bird",
                            "boat", "bottle", "bus", "car", "cat", "chair",
                            "cow", "diningtable", "dog", "horse", "motorbike",
                            "person", "pottedplant", "sheep", "sofa",
                            "train", "tvmonitor"]
        self.colors = np.random.uniform(0, 255,
                                        size=(len(self.class_names), 3))

    def infer(self, frame):
        blob = cv.dnn.blobFromImage(frame, size=(300, 300), swapRB=True,
                                    crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        count = self.count_num_detection(detections)
        return detections, count

    def count_num_detection(self, results):
        count = 0
        for result in results[0, 0]:
            if float(result[2]) > self.confidence_threshold:
                count += 1
        return count