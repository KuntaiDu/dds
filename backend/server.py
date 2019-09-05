import os
import shutil
import logging
import cv2 as cv
from dds_utils import (Results, Region, calc_iou, merge_images,
                       extract_images_from_video)
from .object_detector import Detector


class Server:
    """The server component of DDS protocol. Responsible for running DNN
       on low resolution images, tracking to find regions of interest and
       running DNN on the high resolution regions of interest"""

    def __init__(self, config):
        self.config = config

        self.logger = logging.getLogger("server")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        if not self.config.simulation:
            self.detector = Detector()

        self.logger.info("Server started")

    def perform_detection(self, images_direc, resolution, fnames=None,
                          images=None):
        final_results = Results()
        rpn_regions = Results()

        if fnames is None:
            fnames = sorted(os.listdir(images_direc))
        self.logger.info(f"Running inference on {len(fnames)} frames")
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

            self.logger.debug(f"Running detection for {fname}")
            detection_results, rpn_results = (
                self.detector.infer(image))
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

            if frame_with_no_results:
                final_results.append(
                    Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution))

        return final_results, rpn_regions

    def get_regions_to_query(self, rpn_regions, detections):
        req_regions = Results()
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

    def simulate_low_query(self, start_fid, end_fid, images_direc,
                           results_dict, simulation=True,
                           rpn_enlarge_ratio=0.0):
        base_req_regions = Results()
        for fid in range(start_fid, end_fid):
            base_req_regions.append(
                Region(fid, 0, 0, 1, 1, 1.0, 2,
                       self.config.high_resolution))
        extract_images_from_video(images_direc, base_req_regions)

        batch_results = Results()

        self.logger.info(f"Getting results with threshold "
                         f"{self.config.low_threshold} and "
                         f"{self.config.high_threshold}")
        # Extract relevant results
        for fid in range(start_fid, end_fid):
            fid_results = results_dict[fid]
            for single_result in fid_results:
                single_result.origin = "low-res"
                batch_results.add_single_result(
                    single_result, self.config.intersection_threshold)

        detections = Results()
        rpn_regions = Results()
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

    def emulate_high_query(self, vid_name, low_images_direc, req_regions):
        images_direc = vid_name + "-cropped"
        # Extract images from encoded video
        extract_images_from_video(images_direc, req_regions)

        if not os.path.isdir(images_direc):
            self.logger.error("Images directory was not found but the "
                              "second iteration was called anyway")
            return Results()

        fnames = sorted([f for f in os.listdir(images_direc) if "png" in f])

        # Make seperate directory and copy all images to that directory
        merged_images_direc = os.path.join(images_direc, "merged")
        os.makedirs(merged_images_direc, exist_ok=True)
        for img in fnames:
            shutil.copy(os.path.join(images_direc, img), merged_images_direc)

        merged_images = merge_images(
            merged_images_direc, low_images_direc, req_regions)
        results, _ = self.perform_detection(
            merged_images_direc, self.config.high_resolution, fnames,
            merged_images)

        results_with_detections_only = Results()
        for r in results.regions:
            results_with_detections_only.add_single_result(
                r, self.config.intersection_threshold)
        shutil.rmtree(merged_images_direc)

        return results_with_detections_only
