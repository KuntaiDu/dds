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
import json


class Server:
    """The server component of DDS protocol. Responsible for running DNN
       on low resolution images, tracking to find regions of interest and
       running DNN on the high resolution regions of interest"""

    def __init__(self, config):
        self.config = config

        self.logger = logging.getLogger("server")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        # Initialize a neural network model to be used
        self.model = Model_Creator()(config)

        # Initialize an Application object
        self.app = Application_Creator()(self)

        self.first_phase_folder = 'server_temp_dds'
        self.second_phase_folder = 'server_temp_dds-cropped'

        # clean up existing tmp files
        self.perform_server_cleanup()

        self.logger.info("Server started")

    def reset_state(self):
        self.perform_server_cleanup()

    def perform_server_cleanup(self):
        os.system('rm -r -f server_temp*')


    def extract_video_to_folder(self, folder, start_fid, end_fid):
        feedback_regions = Regions()
        for fid in range(start_fid, end_fid):
            feedback_regions.append(
                Region(fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
        extract_images_from_video(folder, feedback_regions)

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

        # note: This serves as initialization of feedback_regions (feedback for client)
        # Extract images
        # Make req regions for extraction
        # feedback_regions, fnames = self.app.initialize_feedback_regions()
        self.logger.info(f"Processing frames from {start_fid} to {end_fid}")
        self.extract_video_to_folder(self.first_phase_folder, start_fid, end_fid)
        fnames = [f for f in os.listdir(self.first_phase_folder) if "png" in f]

        # generate feedback
        detection_feedback_dic = self.app.run_inference_with_feedback(start_fid, end_fid, self.model, self.first_phase_folder, fnames, self.config)


        return detection_feedback_dic

    def perform_high_query(self, file_data, json_data):

        # dealing with the arguments and paths
        low_images_direc = self.first_phase_folder
        cropped_images_direc = self.second_phase_folder
        Path(cropped_images_direc).mkdir(exist_ok=True)
        with open(os.path.join(cropped_images_direc, "temp.mp4"), "wb") as f:
            f.write(file_data.read())
        images_direc = low_images_direc + "-cropped"
        feedback_regions = Regions(json.load(json_data))

        # Extract images from encoded video
        extract_images_from_video(images_direc, feedback_regions)

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
            merged_images_direc, low_images_direc, feedback_regions)

        # run inference
        inference_results = self.app.run_inference(
                        self.model, merged_images_direc, 
                        self.config.high_resolution, fnames, merged_images, config=self.config)
        results = inference_results["results"]

        shutil.rmtree(merged_images_direc)

        # Perform server side cleanup for the next batch
        self.perform_server_cleanup()

        return {
            "inference_results": results.toJSON()
        }
