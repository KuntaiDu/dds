import os
import shutil
import cv2 as cv
from results.regions import (Regions, Region)
from results.results import Results
from models.object_detector import Detector
from results.regions import (calc_iou, merge_images,
                       extract_images_from_video, merge_boxes_in_results,
                       compute_area_of_frame, calc_area, read_results_dict)
from abc import ABC, abstractmethod


class Application(ABC):
    def __init__(self, config):
        self.config = config # configuration, same as server's
        self.type_app = config["application"] # application type

    # run inference (previously known as perform_detection)
    @abstractmethod
    def run_inference(self, detector, images_direc, resolution, fnames=None, images=None):
        pass

    # create an empty results
    def create_empty_results(self):
        pass

    @abstractmethod
    def run_inference_with_feedback(self, start_fid, end_fid, detector, images_direc, fnames, config):
        pass

    @abstractmethod
    def get_deserializer(self):
        pass

    def postprocess_results(self):
        
        return lambda final_results, number_of_frames: final_results