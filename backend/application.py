import os
import shutil
import cv2 as cv
from classes.regions import (Regions, Region)
from classes.results import Results
from .object_detector import Detector
from classes.regions import (calc_iou, merge_images,
                       extract_images_from_video, merge_boxes_in_results,
                       compute_area_of_frame, calc_area, read_results_dict)

class Application:
    def __init__(self, config):
        self.config = config # configuration, same as server's
        self.type_app = config["application"] # application type

    # return an object of a child class of Results() based on application type
    def create_empty_results(self):
        if self.type_app == 'object_detection':
            return Regions()
        elif self.type_app == 'semantic_segmentation':
            return None # under construction

    # run inference (previously known as perform_detection)
    def run_inference(self, detector, images_direc, resolution, fnames=None, images=None):
        # object_detection
        if self.type_app == 'object_detection': 
            final_results = Regions() # results in the form of regions
            rpn_regions = Regions()

            if fnames is None:
                fnames = sorted(os.listdir(images_direc))
            # self.logger.info(f"Running inference on {len(fnames)} frames")
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

                detection_results, rpn_results = (
                    detector.infer(image))
                frame_with_no_results = True
                for label, conf, (x, y, w, h) in detection_results:
                    if (self.config.min_object_size and
                            w * h < self.config.min_object_size) or w * h == 0.0:
                        continue
                    r = Region(fid, x, y, w, h, conf, label,
                            resolution, origin="mpeg")
                    final_results.append(r)
                    frame_with_no_results = False
                for label, conf, (x, y, w, h) in rpn_results:
                    r = Region(fid, x, y, w, h, conf, label,
                            resolution, origin="generic")
                    rpn_regions.append(r)
                    frame_with_no_results = False
                #self.logger.debug(
                #    f"Got {len(final_results)} results "
                #    f"and {len(rpn_regions)} for {fname}")

                if frame_with_no_results:
                    final_results.append(
                        Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution))

            # return final_results, rpn_regions
            return {
                "results": final_results,
                "rpn_regions": rpn_regions
            }

        # semantic_segmentation
        elif self.type_app == 'semantic_segmentation':
            print("semantic_segmentation under construction")

    # (Stream A) drive function for generating feedback
    def generate_feedback(self, start_fid, end_fid, images_direc,
                           results, simulation=True,
                           rpn_enlarge_ratio=0.0, extract_regions=True):
        # note that rpn_enlarge_ratio is specific to object detection

        # object_detection
        if self.type_app == 'object_detection': 
            if extract_regions:
                # If called from actual implementation
                # This will not run
                base_req_regions = Regions()
                for fid in range(start_fid, end_fid):
                    base_req_regions.append(
                        Region(fid, 0, 0, 1, 1, 1.0, 2,
                            self.config.high_resolution))
                extract_images_from_video(images_direc, base_req_regions)
            
            batch_results = Regions()
            results_dict = results.regions_dict

            # Extract relevant results
            for fid in range(start_fid, end_fid):
                fid_results = results_dict[fid]
                for single_result in fid_results:
                    single_result.origin = "low-res"
                    batch_results.add_single_result(
                        single_result, self.config.intersection_threshold)

            detections = Regions()
            rpn_regions = Regions()
            # Divide RPN results into detections and RPN regions
            for single_result in batch_results.regions:
                if (single_result.conf > self.config.prune_score and
                        single_result.label == "vehicle"):
                    detections.add_single_result(
                        single_result, self.config.intersection_threshold)
                else:
                    rpn_regions.add_single_result(
                        single_result, self.config.intersection_threshold)

            regions_to_query = self.get_regions_to_query(rpn_regions, detections)

            return detections, regions_to_query
        
        # semantic_segmentation
        elif self.type_app == 'semantic_segmentation':
            print("semantic_segmentation under construction")

    # (Stream A) generate feedback regions based on detections
    def get_regions_to_query(self, regions, detections):
        # note that regions is specific to object detection, need to change a name later
        # detections is determined by Application, can be Regions(), Classes(), etc.

        # object_detection
        if self.type_app == 'object_detection': 
            rpn_regions = regions # change variable name later
            req_regions = Regions()
            for region in rpn_regions.regions:
                # Continue if the size of region is too large
                if region.w * region.h > self.config.size_obj:
                    continue

                # If there are positive detections and they match a region
                # skip that region
                if len(detections) > 0:
                    matches = 0
                    for detection in detections.regions:
                        if (calc_iou(detection, region) >
                                self.config.objfilter_iou and
                                detection.fid == region.fid and
                                region.label == 'object'):
                            matches += 1
                    if matches > 0:
                        continue

                # Enlarge and add to regions to be queried
                region.enlarge(self.config.rpn_enlarge_ratio)
                req_regions.add_single_result(
                    region, self.config.intersection_threshold)
            return req_regions

        # semantic_segmentation
        elif self.type_app == 'semantic_segmentation':
            print("semantic_segmentation under construction")
    
    # (Stream A) combine detection results and feedback regions in a dictionary
    # and send the dic back to the client through network
    def combine_feedback(self, detections, regions_to_query):
        # object_detection
        if self.type_app == 'object_detection': 
            detections_list = []
            for r in detections.regions:
                detections_list.append(
                    [r.fid, r.x, r.y, r.w, r.h, r.conf, r.label])
            req_regions_list = []
            for r in regions_to_query.regions:
                req_regions_list.append(
                    [r.fid, r.x, r.y, r.w, r.h, r.conf, r.label])

            return {
                "results": detections_list,
                "req_regions": req_regions_list
            }

        # semantic_segmentation
        elif self.type_app == 'semantic_segmentation':
            print("semantic_segmentation under construction")

    # (Stream B) generate final results with detections only
    def generate_results_with_detections_only(self, results):
        results_with_detections_only = self.create_empty_results()

        # object_detection
        if self.type_app == 'object_detection': 
            for r in results.regions:
                if r.label == "no obj":
                    continue
                results_with_detections_only.add_single_result(
                    r, self.config.intersection_threshold)

        # semantic_segmentation
        elif self.type_app == 'semantic_segmentation':
            print("semantic_segmentation under construction")

        return results_with_detections_only

    # (Stream B) generate results just from the high query
    def generate_high_only_results(self, results_with_detections_only, req_regions):
        high_only_results = self.create_empty_results()
        area_dict = {}

        # object_detection
        if self.type_app == 'object_detection':
            for r in results_with_detections_only.regions:
                frame_regions = req_regions.regions_dict[r.fid]
                regions_area = 0
                if r.fid in area_dict:
                    regions_area = area_dict[r.fid]
                else:
                    regions_area = compute_area_of_frame(frame_regions)
                    area_dict[r.fid] = regions_area
                regions_with_result = frame_regions + [r]
                total_area = compute_area_of_frame(regions_with_result)
                extra_area = total_area - regions_area
                if extra_area < 0.05 * calc_area(r):
                    r.origin = "high-res"
                    high_only_results.append(r)

        # semantic_segmentation
        elif self.type_app == 'semantic_segmentation':
            print("semantic_segmentation under construction")

        return high_only_results

    # (Stream B) combine final results in a dictionary to send back
    def generate_final_results(self, results):
        results_list = []

        # object_detection
        if self.type_app == 'object_detection':
            for r in results.regions:
                results_list.append([r.fid, r.x, r.y, r.w, r.h, r.conf, r.label])
        
        # semantic_segmentation
        elif self.type_app == 'semantic_segmentation':
            print("semantic_segmentation under construction")

        return results_list