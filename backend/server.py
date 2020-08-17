import os
import shutil
import logging
import cv2 as cv
from results.regions import (Regions, Region)
from results.results import Results
from results.regions import (calc_iou, merge_images,
                       extract_images_from_video, merge_boxes_in_results,
                       compute_area_of_frame, calc_area, read_results_dict)
from models.model_creator import Model_Creator
from application.application_creator import Application_Creator
from pathlib import Path


class Server:
    """The server component of DDS protocol. Responsible for running DNN
       on low resolution images, tracking to find regions of interest and
       running DNN on the high resolution regions of interest"""

    def __init__(self, config, nframes=None):
        self.config = config

        self.logger = logging.getLogger("server")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        # Initialize a neural network model to be used
        self.model_creator = Model_Creator()
        mod_creator_functions = {
            'object_detection': self.model_creator.create_object_detector}
        self.model = mod_creator_functions[config['application']]()

        # Initialize an Application object
        self.app_creator = Application_Creator()
        app_creator_functions = {'object_detection': self.app_creator.create_object_detection}
        self.app = app_creator_functions[config['application']](config)

        self.curr_fid = 0
        self.nframes = nframes
        self.last_requested_regions = None
        self.first_phase_folder = 'server_temp_dds'
        self.second_phase_folder = 'server_temp_dds-cropped'

        self.logger.info("Server started")

    def reset_state(self, nframes):
        self.curr_fid = 0
        self.nframes = nframes
        self.last_requested_regions = None
        self.perform_server_cleanup()

    def perform_server_cleanup(self):
        os.system('rm -r -f server_temp*')

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
                        self.model, merged_images_direc, 
                        self.config.high_resolution, fnames, merged_images)
        results = inference_results["results"]

        # generate results_with_detections_only
        results_with_detections_only = self.app.generate_results_with_detections_only(results)

        # generate results just from the high query
        high_only_results = self.app.generate_high_only_results(
                                results_with_detections_only, req_regions)

        shutil.rmtree(merged_images_direc)

        return results_with_detections_only

    def extract_video_to_folder(self, folder, start_fid, end_fid):
        req_regions = Regions()
        for fid in range(start_fid, end_fid):
            req_regions.append(
                Region(fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
        extract_images_from_video(folder, req_regions)

    def run_inference(self, start_fid, end_fid, video_data):
        # Write in-memory video to file
        video_folder = Path('server_temp_inference')
        video_folder.mkdir(exist_ok = True)
        with open(video_folder / 'temp.mp4', 'wb') as f:
            f.write(video_data.read())

        self.logger.info(f"Processing frames from {start_fid} to {end_fid}")
        self.extract_video_to_folder(str(video_folder), 0, self.config.batch_size)
        
    def perform_low_query(self, start_fid, end_fid, vid_data):
        # Write video to file
        Path(self.first_phase_folder).mkdir(exist_ok=True)
        with open(os.path.join(self.first_phase_folder, "temp.mp4"), "wb") as f:
            f.write(vid_data.read())

        # note: This serves as initialization of req_regions (feedback for client)
        # Extract images
        # Make req regions for extraction
        # req_regions, fnames = self.app.initialize_req_regions()
        self.logger.info(f"Processing frames from {start_fid} to {end_fid}")
        self.extract_video_to_folder(self.first_phase_folder, start_fid, end_fid)
        fnames = [f for f in os.listdir(self.first_phase_folder) if "png" in f]

        # generate feedback
        detection_feedback_dic, feedback = self.app.run_inference_with_feedback(start_fid, end_fid, self.model, self.first_phase_folder, fnames, self.config)

        self.last_requested_regions = feedback
        self.curr_fid = end_fid

        return detection_feedback_dic

    def perform_high_query(self, file_data):
        low_images_direc = self.first_phase_folder
        cropped_images_direc = self.second_phase_folder
        Path(cropped_images_direc).mkdir(exist_ok=True)
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
            "inference_results": results_list,
        }
