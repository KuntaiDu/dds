import os
import shutil
import cv2 as cv
from results.regions import (Regions, Region)
from results.results import Results
from backend.object_detector import Detector
from results.regions import (calc_iou, merge_images,
                       extract_images_from_video, merge_boxes_in_results,
                       compute_area_of_frame, calc_area, read_results_dict)
from abc import abstractmethod


class Application:
    def __init__(self, config):
        self.config = config # configuration, same as server's
        self.type_app = config["application"] # application type

    # return an object of a child class of Results() based on application type
    @abstractmethod
    def create_empty_results(self):
        pass

    # run inference (previously known as perform_detection)
    @abstractmethod
    def run_inference(self, detector, images_direc, resolution, fnames=None, images=None):
        pass

    # (Stream A) drive function for generating feedback
    @abstractmethod
    def generate_feedback(self, start_fid, end_fid, images_direc,
                           results, simulation=True,
                           rpn_enlarge_ratio=0.0, extract_regions=True):
        # note that rpn_enlarge_ratio is specific to object detection
        pass

    # (Stream A) generate feedback regions based on detections
    @abstractmethod
    def get_regions_to_query(self, regions, detections):
        # note that regions is specific to object detection, need to change a name later
        # detections is determined by Application, can be Regions(), Classes(), etc.
        pass
    
    # (Stream A) combine detection results and feedback regions in a dictionary
    # and send the dic back to the client through network
    @abstractmethod
    def combine_feedback(self, detections, regions_to_query):
        pass

    # (Stream B) generate final results with detections only
    @abstractmethod
    def generate_results_with_detections_only(self, results):
        pass

    # (Stream B) generate results just from the high query
    @abstractmethod
    def generate_high_only_results(self, results_with_detections_only, req_regions):
        pass

    # (Stream B) combine final results in a dictionary to send back
    @abstractmethod
    def generate_final_results(self, results):
        pass