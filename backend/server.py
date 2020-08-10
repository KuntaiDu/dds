import os
import shutil
import logging
import cv2 as cv
from results.regions import (Regions, Region)
from results.results import Results
from results.regions import (calc_iou, merge_images,
                       extract_images_from_video, merge_boxes_in_results,
                       compute_area_of_frame, calc_area, read_results_dict)
from .object_detector import Detector
from application.application_creator import Application_Creator


class Server:
    """The server component of DDS protocol. Responsible for running DNN
       on low resolution images, tracking to find regions of interest and
       running DNN on the high resolution regions of interest"""

    def __init__(self, config, nframes=None):
        self.config = config

        self.logger = logging.getLogger("server")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        # Initialize a Dector object
        self.detector = Detector()

        # Initialize an Application object
        self.app_creator = Application_Creator()
        dic_creator_functions = {'object_detection': self.app_creator.create_object_detection}
        self.app = dic_creator_functions[config['application']](config)

        self.curr_fid = 0
        self.nframes = nframes
        self.last_requested_regions = None

        self.logger.info("Server started")

    def reset_state(self, nframes):
        self.curr_fid = 0
        self.nframes = nframes
        self.last_requested_regions = None
        for f in os.listdir("server_temp"):
            os.remove(os.path.join("server_temp", f))
        for f in os.listdir("server_temp-cropped"):
            os.remove(os.path.join("server_temp-cropped", f))

    def perform_server_cleanup(self):
        for f in os.listdir("server_temp"):
            os.remove(os.path.join("server_temp", f))
        for f in os.listdir("server_temp-cropped"):
            os.remove(os.path.join("server_temp-cropped", f))

    def emulate_high_query(self, vid_name, low_images_direc, req_regions):
        images_direc = vid_name + "-cropped"
        # Extract images from encoded video
        extract_images_from_video(images_direc, req_regions)

        if not os.path.isdir(images_direc):
            self.logger.error("Images directory was not found but the "
                              "second iteration was called anyway")
            return Regions()

        fnames = sorted([f for f in os.listdir(images_direc) if "png" in f])

        # Make seperate directory and copy all images to that directory
        merged_images_direc = os.path.join(images_direc, "merged")
        os.makedirs(merged_images_direc, exist_ok=True)
        for img in fnames:
            shutil.copy(os.path.join(images_direc, img), merged_images_direc)

        merged_images = merge_images(
            merged_images_direc, low_images_direc, req_regions)

        # (modified) run inference
        inference_results = self.app.run_inference(
                        self.detector, merged_images_direc, 
                        self.config.high_resolution, fnames, merged_images)
        results = inference_results["results"]

        # generate results_with_detections_only
        results_with_detections_only = self.app.generate_results_with_detections_only(results)

        # generate results just from the high query
        high_only_results = self.app.generate_high_only_results(
                                results_with_detections_only, req_regions)

        shutil.rmtree(merged_images_direc)

        return results_with_detections_only

    def perform_low_query(self, vid_data):
        # Write video to file
        with open(os.path.join("server_temp", "temp.mp4"), "wb") as f:
            f.write(vid_data.read())

        # note: This serves as initialization of req_regions (feedback for client)
        # Extract images
        # Make req regions for extraction
        # req_regions, fnames = self.app.initialize_req_regions()
        start_fid = self.curr_fid
        end_fid = min(self.curr_fid + self.config.batch_size, self.nframes)
        self.logger.info(f"Processing frames from {start_fid} to {end_fid}")
        req_regions = Regions()
        for fid in range(start_fid, end_fid):
            req_regions.append(
                Region(fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
        extract_images_from_video("server_temp", req_regions)
        fnames = [f for f in os.listdir("server_temp") if "png" in f]

        # results for current batch of images
        # WARNING: there might be some issue here!
        batch_results = self.app.create_empty_results() 

        # run inference
        inference_results = self.app.run_inference(self.detector, "server_temp",
                                            self.config.low_resolution, fnames)
        results = inference_results["results"]

        # combine these results to final_results
        batch_results.combine_results(results, self.config.intersection_threshold)

        # note: in fact using ifs here is not the optimal design
        #       but currently I have no better idea
        #       since merging bounding boxes is specific to object detection
        if self.app.type_app == 'object_detection': # object_detection
            # need to merge this because all previous experiments assumed
            # that low (mpeg) results are already merged
            rpn_regions = inference_results["rpn_regions"]
            batch_results = merge_boxes_in_results(
                batch_results.regions_dict, 0.3, 0.3)
            # combine results in rpn regions
            batch_results.combine_results(rpn_regions, self.config.intersection_threshold)

        # let Application object generate feedback (detections results and feedback regions)
        # ideally, generate_feedback should replace simulate_low_query
        # detections here is an object of a child class of Results, determined by Application
        # e.g. for object detection, detections is Regions()
        # note that we originally used simulate_low_query for this part
        detections, regions_to_query = self.app.generate_feedback(
            start_fid, end_fid, "server_temp", batch_results,
            False, self.config.rpn_enlarge_ratio, False)

        self.last_requested_regions = regions_to_query
        self.curr_fid = end_fid

        # Make dictionary to be sent back
        feedback_dic = self.app.combine_feedback(detections, regions_to_query)
        
        return feedback_dic # this contains detection results and feedback regions

    def perform_high_query(self, file_data):
        low_images_direc = "server_temp"
        cropped_images_direc = "server_temp-cropped"

        with open(os.path.join(cropped_images_direc, "temp.mp4"), "wb") as f:
            f.write(file_data.read())

        # results here can be Regions(), Classes(), etc.
        results = self.emulate_high_query(
            low_images_direc, low_images_direc, self.last_requested_regions)

        # generate final result to send back to client
        results_list = self.app.generate_final_results(results)

        # Perform server side cleanup for the next batch
        self.perform_server_cleanup()

        return {
            "results": results_list,
            "req_region": []
        }
