import logging
import os
import shutil
import requests
import json
from results.regions import (Regions, Region)
from results.regions import (read_results_dict, cleanup,
                       compute_regions_size, extract_images_from_video,
                       merge_boxes_in_results)
from application.application_creator import Application_Creator
import yaml

class Client:
    """The client of the DDS protocol
       sends images in low resolution and waits for
       further instructions from the server. And finally receives results
       Note: All frame ranges are half open ranges"""

    def __init__(self, hname, config, server_handle=None):
        if hname:
            self.hname = hname
            self.session = requests.Session()
        else:
            self.server = server_handle
        self.config = config

        # Initialize an Application object
        self.app = Application_Creator()(self)

        self.logger = logging.getLogger("client")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.logger.info(f"Client initialized")

    def init_server(self, nframes):
        response = self.session.post(
            "http://" + self.hname + "/initialize_server", data=yaml.dump(self.config))
        if response.status_code != 200:
            self.logger.fatal("Could not initialize server")
            # Need to add exception handling
            exit()

    def post_video_to_server(self, vid_name, function_name, deserializer, json_object=None, **kwargs):

        # stream to the server
        encoded_vid_path = os.path.join(
            vid_name, "temp.mp4")
        video_to_send = {"media": open(encoded_vid_path, "rb"), "json": json.dumps(json_object)}
        response = self.session.post(
            "http://" + self.hname + f"/{function_name}", files=video_to_send, params=kwargs)
        response_json = json.loads(response.text)

        # deserialize from json results
        for key in deserializer.keys():
            response_json[key] = deserializer[key](response_json[key])

        return response_json


    def get_first_phase_results(self, vid_name, start_fid, end_fid):

        deserializer_results = lambda x: Regions(x)
        deserializer_feedbacks = lambda x: Regions(x)

        deserializer = {
            'inference_results': deserializer_results,
            'feedback_regions': deserializer_feedbacks
        }

        response_json = self.post_video_to_server(vid_name + "-base-phase-cropped", 'perform_low_query', deserializer, start_fid = start_fid, end_fid = end_fid)

        return response_json['inference_results'], response_json['feedback_regions']

    def get_second_phase_results(self, vid_name, feedback):

        deserializer = {'inference_results': lambda x: Regions(x)}

        response_json = self.post_video_to_server(vid_name + "-cropped", 'perform_high_query', deserializer, feedback.toJSON())

        return response_json['inference_results']

    def analyze_video(
            self, vid_name, raw_images, config, enforce_iframes):
        final_results = self.app.create_empty_results()
        all_required_regions = Regions()
        low_phase_size = 0
        high_phase_size = 0
        nframes = sum(map(lambda e: "png" in e, os.listdir(raw_images)))

        # initialize server
        self.init_server(nframes)

        for i in range(0, nframes, self.config.batch_size):
            start_frame = i
            end_frame = min(nframes, i + self.config.batch_size)
            self.logger.info(f"Processing frames {start_frame} to {end_frame}")

            # First iteration
            req_regions = Regions()
            for fid in range(start_frame, end_frame):
                req_regions.append(Region(
                    fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{vid_name}-base-phase", raw_images,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            low_phase_size += batch_video_size
            self.logger.info(f"{batch_video_size // 1024}KB sent in base phase."
                             f" Using QP {self.config.low_qp} and "
                             f"resolution {self.config.low_resolution}.")

            # stream to the server
            results, feedback_regions = self.get_first_phase_results(vid_name, start_frame, end_frame)

            final_results.combine_results(
                results, self.config.intersection_threshold)
            all_required_regions.combine_results(
                feedback_regions, self.config.intersection_threshold)

            # Second Iteration
            if len(feedback_regions) > 0:
                batch_video_size, _ = compute_regions_size(
                    feedback_regions, vid_name, raw_images,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                high_phase_size += batch_video_size
                self.logger.info(f"{batch_video_size // 1024}KB sent in second "
                                 f"phase. Using QP {self.config.high_qp} and "
                                 f"resolution {self.config.high_resolution}.")
                results = self.get_second_phase_results(vid_name, feedback_regions)
                final_results.combine_results(
                    results, self.config.intersection_threshold)

            # Cleanup for the next batch
            cleanup(vid_name, False, start_frame, end_frame)

        self.logger.info(f"Merging results")
        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        self.logger.info(f"Writing results for {vid_name}")
        final_results.fill_gaps(nframes)

        final_results.combine_results(
            all_required_regions, self.config.intersection_threshold)

        final_results.write(f"{vid_name}")

        return final_results, (low_phase_size, high_phase_size)

        
    def analyze_video_mpeg(self, video_name, raw_images_path, enforce_iframes):
        number_of_frames = len(
            [f for f in os.listdir(raw_images_path) if ".png" in f])

        final_results = self.app.create_empty_results()
        final_feedback_regions = Regions()
        total_size = 0
        for i in range(0, number_of_frames, self.config.batch_size):
            start_frame = i
            end_frame = min(number_of_frames, i + self.config.batch_size)

            batch_fnames = sorted([f"{str(idx).zfill(10)}.png"
                                   for idx in range(start_frame, end_frame)])

            req_regions = Regions()
            for fid in range(start_frame, end_frame):
                req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.low_resolution))
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{video_name}-base-phase", raw_images_path,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"{batch_video_size / 1024}KB sent "
                        f"in base phase using {self.config.low_qp}QP")
            extract_images_from_video(f"{video_name}-base-phase-cropped",
                                      req_regions)
            results_dict = (
                self.server.app.run_inference( 
                    self.server.model,
                    f"{video_name}-base-phase-cropped",
                    self.config.low_resolution, batch_fnames)) # perviously perform_detection
            results = results_dict["results"]

            self.logger.info(f"Processed batch {start_frame} to {end_frame} with a "
                        f"total video size of {batch_video_size / 1024}KB")
            final_results.combine_results(
                results, self.config.intersection_threshold)

            # Remove encoded video manually
            shutil.rmtree(f"{video_name}-base-phase-cropped")
            total_size += batch_video_size

        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        final_results.fill_gaps(number_of_frames)

        final_results.write(video_name)

        return final_results, [total_size, 0]
