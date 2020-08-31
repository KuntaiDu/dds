import os
import shutil
import cv2 as cv
from results.regions import (Regions, Region)
from results.results import Results
from models.object_detector import Detector
from results.regions import (calc_iou, merge_images,
                       extract_images_from_video, merge_boxes_in_results,
                       compute_area_of_frame, calc_area, read_results_dict, 
                       dict_to_list)
from .application import Application
import logging


class Object_Detection(Application):
    def __init__(self, server):
        self.server = server
        self.type_app = "object_detection" # application type
        self.logger = logging.getLogger("object_detection")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

    # return an object of a child class of Results() based on application type
    def create_empty_results(self):
        return Regions()

    def get_deserializer(self):
        return lambda x: Regions(x)

    # run inference (previously known as perform_detection)
    def run_inference(self, detector, images_direc, resolution, fnames=None, images=None):
        final_results = Regions() # results in the form of regions
        rpn_regions = Regions()

        if fnames is None:
            fnames = sorted(os.listdir(images_direc))
        self.logger.info(f"Running object detection on {len(fnames)} frames")
        for fname in fnames:

            if "png" not in fname:
                continue

            fid = int(fname.split(".")[0])
            image = None
            if images:
                image = images[fid]
            else:
                image_path = os.path.join(images_direc, fname)
                image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            # perform inference
            detection_results, rpn_results = (
                detector.infer(image))

            # filter those strange objects
            frame_with_no_results = True
            for label, conf, (x, y, w, h) in detection_results:
                if w * h == 0.0:
                    continue
                r = Region(fid, x, y, w, h, conf, label,
                        resolution, "mpeg")
                final_results.append(r)
                frame_with_no_results = False
            for label, conf, (x, y, w, h) in rpn_results:
                r = Region(fid, x, y, w, h, conf, label,
                        resolution, "generic")
                rpn_regions.append(r)
                frame_with_no_results = False

            if frame_with_no_results:
                final_results.append(
                    Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution))

        # return final_results, rpn_regions
        return {
            "results": final_results,
            "feedback_regions": rpn_regions
        }

    # run inference and generate feedback regions
    def run_inference_with_feedback(self, start_fid, end_fid, detector, images_direc, fnames, config):
        
        # run object detection
        results = self.run_inference(detector, images_direc, config.low_resolution, fnames)

        # merge several results
        merged_results = self.create_empty_results()
        merged_results.combine_results(results['results'], config.intersection_threshold)
        merged_results = merge_boxes_in_results(merged_results.regions_dict, 0.3, 0.3)
        merged_results.combine_results(results['feedback_regions'], config.intersection_threshold)

        # get feedback regions
        detection, feedback = self.generate_feedback(start_fid, end_fid, images_direc, merged_results.regions_dict, False, config.rpn_enlarge_ratio, False)


        return {
            "inference_results": detection.toJSON(),
            "feedback_regions": feedback.toJSON()
        }



    # (Stream A) drive function for generating feedback
    def generate_feedback(self, start_fid, end_fid, images_direc,
                           results_dict, simulation=True,
                           rpn_enlarge_ratio=0.0, extract_regions=True):
        # note that rpn_enlarge_ratio is specific to object detection

        if extract_regions:
            # If called from actual implementation
            # This will not run
            base_req_regions = Regions()
            for fid in range(start_fid, end_fid):
                base_req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                        self.server.config.low_resolution))
            extract_images_from_video(images_direc, base_req_regions)
        
        batch_results = Regions()

        # Extract relevant results
        for fid in range(start_fid, end_fid):
            fid_results = results_dict[fid]
            for single_result in fid_results:
                single_result.origin = "low-res"
                batch_results.add_single_result(
                    single_result, self.server.config.intersection_threshold)

        detections = Regions()
        rpn_regions = Regions()
        # Divide RPN results into detections and RPN regions
        for fid in batch_results.regions_dict:
            for single_result in batch_results.regions_dict[fid]:
                if single_result.label == "vehicle":
                    # these are detection results
                    detections.add_single_result(
                        single_result, self.server.config.intersection_threshold)
        
                rpn_regions.add_single_result(
                    single_result, self.server.config.intersection_threshold)

        regions_to_query = self.get_regions_to_query(rpn_regions, detections)

        return detections, regions_to_query

    # (Stream A) generate feedback regions based on detections
    def get_regions_to_query(self, regions, detections):
        # note that regions is specific to object detection, need to change a name later
        # detections is determined by Application, can be Regions(), Classes(), etc.

        rpn_regions = regions # change variable name later
        req_regions = Regions()

        rpn_regions_list = dict_to_list(rpn_regions.regions_dict)
        for region in rpn_regions_list:
            # Continue if the size of region is too large
            if region.w * region.h > self.server.config.size_obj:
                continue

            # If there are positive detections and they match a region
            # skip that region
            if len(detections) > 0:
                matches = 0
                detection_regions_list = dict_to_list(detections.regions_dict)
                for detection in detection_regions_list:
                    if (calc_iou(detection, region) >
                            self.server.config.objfilter_iou and
                            detection.fid == region.fid and
                            region.label == 'object'):
                        matches += 1
                if matches > 0:
                    continue

            # Enlarge and add to regions to be queried
            region.enlarge(self.server.config.rpn_enlarge_ratio)
            req_regions.add_single_result(
                region, self.server.config.intersection_threshold)
        return req_regions
        
    def postprocess_results(self):
        
        def helper(self, results, number_of_frames):
            
            final_results = merge_boxes_in_results(results.regions_dict, 0.3, 0.3)
            final_results.fill_gaps(number_of_frames)

            return final_results

        return helper
