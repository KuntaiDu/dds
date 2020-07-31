import re
import os
import csv
import shutil
import subprocess
import numpy as np
import cv2 as cv
import networkx
from networkx.algorithms.components.connected import connected_components
from results.regions import (Regions, Region)


class ServerConfig:
    def __init__(self, low_res, high_res, low_qp, high_qp, bsize,
                 h_thres, l_thres, max_obj_size, min_obj_size,
                 tracker_length, boundary, intersection_threshold,
                 tracking_threshold, suppression_threshold, simulation,
                 rpn_enlarge_ratio, prune_score, objfilter_iou, size_obj):
        self.low_resolution = low_res
        self.high_resolution = high_res
        self.low_qp = low_qp
        self.high_qp = high_qp
        self.batch_size = bsize
        self.high_threshold = h_thres
        self.low_threshold = l_thres
        self.max_object_size = max_obj_size
        self.min_object_size = min_obj_size
        self.tracker_length = tracker_length
        self.boundary = boundary
        self.intersection_threshold = intersection_threshold
        self.simulation = simulation
        self.tracking_threshold = tracking_threshold
        self.suppression_threshold = suppression_threshold
        self.rpn_enlarge_ratio = rpn_enlarge_ratio
        self.prune_score = prune_score
        self.objfilter_iou = objfilter_iou
        self.size_obj = size_obj