import logging
import os
import cv2 as cv
import shutil
import time
import requests
import json
from .local_object_detector import LocalObjectDetector
from results.regions import (Regions, Region)
from results.regions import (read_results_dict, cleanup,
                       compute_regions_size, extract_images_from_video,
                       merge_boxes_in_results, calc_frame_difference)
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
        self.deserializer = self.app.get_deserializer()

        self.logger = logging.getLogger("client")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.trackers = {}

        self.logger.info(f"Client initialized")

    def init_server(self):
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


        deserializer = {
            'inference_results': self.deserializer,
            'feedback_regions': lambda x: Regions(x)
        }

        response_json = self.post_video_to_server(vid_name + "-base-phase-cropped", 'perform_low_query', deserializer, start_fid = start_fid, end_fid = end_fid)

        return response_json['inference_results'], response_json['feedback_regions']

    def get_second_phase_results(self, vid_name, feedback):

        deserializer = {'inference_results': self.deserializer}

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
        self.init_server()

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
            self.logger.info("Contain {} feedback regions".format(len(feedback_regions)))
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

        final_results = self.app.postprocess_results()(final_results, nframes)

        final_results.combine_results(
            all_required_regions, self.config.intersection_threshold)

        final_results.write(f"{vid_name}")

        return final_results, (low_phase_size, high_phase_size)

        
    def analyze_video_mpeg(self, video_name, raw_images_path, enforce_iframes):        
        number_of_frames = len(
            [f for f in os.listdir(raw_images_path) if ".png" in f])
        final_results = self.app.create_empty_results()
        total_size = 0

        # initialize server
        self.init_server()

        for i in range(0, number_of_frames, self.config.batch_size):
            start_frame = i
            end_frame = min(number_of_frames, i + self.config.batch_size)

            req_regions = Regions()
            for fid in range(start_frame, end_frame):
                req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.low_resolution))
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{video_name}-base-phase", raw_images_path,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"{batch_video_size // 1024}KB sent "
                        f"in base phase using {self.config.low_qp}QP")
            results, _ = self.get_first_phase_results(video_name, start_frame, end_frame)

            self.logger.info(f"Processed batch {start_frame} to {end_frame} with a "
                        f"total video size of {batch_video_size / 1024}KB")
            final_results.combine_results(
                results, self.config)

            # Remove encoded video manually
            shutil.rmtree(f"{video_name}-base-phase-cropped")
            total_size += batch_video_size

        final_results = self.app.postprocess_results()(final_results, number_of_frames)

        final_results.write(video_name)

        return final_results, [total_size, 0]


    def initialize_trackers(self, results, fid, images_path):
        self.trackers = {}
        init_fname = f"{str(fid).zfill(10)}.png"
        init_frame_path = os.path.join(images_path, init_fname)
        init_frame = cv.imread(init_frame_path)
        im_height, im_width, _ = init_frame.shape
        for idx, r in enumerate(results.regions_dict[fid]):
            tracker = cv.TrackerKCF_create()
            self.trackers[f"{fid}_{idx}"] = tracker
            x = r.x * im_width
            y = r.y * im_height
            w = r.w * im_width
            h = r.h * im_height
            bbox = (x, y, w, h)
            tracker.init(init_frame, bbox)

    def is_trigger_frame_glimpse(self, last_sent, curr_frame):
        if last_sent is None:
            return True
        width = last_sent.shape[1]
        height = last_sent.shape[0]
        diff = calc_frame_difference(last_sent, curr_frame)
        print(diff, (width * height) / 2)
        if diff > (width * height) / 2:
            return True
        else:
            return False
    

    def track(self, final_results, curr_fid, images_path):
        tracking_results = self.app.create_empty_results()
        tracking_success = True

        curr_fname = f"{str(curr_fid).zfill(10)}.png"
        curr_frame_path = os.path.join(images_path, curr_fname)
        curr_frame = cv.imread(curr_frame_path)

        to_remove = []
        im_height, im_width, _ = curr_frame.shape
        for key, tracker in self.trackers.items():
            fid, idx = [int(e) for e in key.split("_")]
            obj = final_results.regions_dict[fid][idx]
            # Initialize using prev frame
            t = time.time()
            status, bbox = tracker.update(curr_frame)
            tracking_time = time.time() - t
            if tracking_time > 1.5:
                to_remove.append(key)
            tracking_success = tracking_success & status
            if status:
                x = bbox[0] / im_width
                y = bbox[1] / im_height
                w = bbox[2] / im_width
                h = bbox[3] / im_height
                region = Region(curr_fid, x, y, w, h,
                                obj.conf, obj.label,
                                obj.resolution, "tracking")
                tracking_results.append(region)
            else:
                to_remove.append(key)
        for k in to_remove:
            if k in self.trackers:
                del self.trackers[k]

        return tracking_results, tracking_success
    

    # Currently only supports object detection
    def analyze_video_glimpse_emulate(self, video_name, images_path,
                                      im_width=640, im_height=480):
        number_of_frames = len(
            [e for e in os.listdir(images_path) if "png" in e])
        last_sent = None
        final_results = Regions()
        tracking_success = True
        total_sent = 0
        last_sent_id = None
        for fid in range(0, number_of_frames):
            fname = f"{str(fid).zfill(10)}.png"
            frame_path = os.path.join(images_path, fname)
            curr_frame = cv.imread(frame_path)
            curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
            if (self.is_trigger_frame_glimpse(
                    last_sent, curr_gray) or not tracking_success) and (
                        not last_sent_id or (fid - last_sent_id > 5)):
                resized_frame = cv.resize(curr_frame, (im_height, im_width))
                self.logger.info(f"Sending frame to server {fid}")
                cv.imwrite(f"{video_name}_temp.jpg", resized_frame,
                           [cv.IMWRITE_JPEG_QUALITY, 70])
                resized_frame = cv.imread(f"{video_name}_temp.jpg")

                # run inference
                results = self.app.run_inference(
                    self.server.model, None, 1.0, [fname], {fid: resized_frame}, config=self.config)
                detection_results = results["results"]
                
                for fid in detection_results.regions_dict:
                    for r in detection_results.regions_dict[fid]:
                        r.origin = "glimpse-detection"
                last_sent = curr_gray
                final_results.combine_results(detection_results, 1.0)
                last_sent_id = fid
                tracking_success = True
                # WARNING: a bug here when combining results
                # Some regions are not combined since exisiting results are better
                # This leads to an index out of range error
                #self.initialize_trackers(detection_results, fid, images_path)
                self.initialize_trackers(final_results, fid, images_path)
                # for debugging purposes
                #print(len(detection_results.regions_dict[fid]))
                #print(len(final_results.regions_dict[fid]))

                total_sent += os.path.getsize(f"{video_name}_temp.jpg")
                os.remove(f"{video_name}_temp.jpg")

            # Pick objects in last frame and perform tracking in current frame
            if fid > 0 and fid != last_sent_id:
                tracking_results, tracking_success = self.track(
                    final_results, fid, images_path)
                self.logger.info(f"Got {len(tracking_results)} from tracking frame {fid}")
                for fid in tracking_results.regions_dict:
                    for r in tracking_results.regions_dict[fid]:
                        r.origin = "glimpse-tracking"
                final_results.combine_results(tracking_results, 1.0)

        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        final_results.write(video_name)

        return final_results, [total_sent, 0]


    # Currently only supports object detection
    def analyze_video_glimpse(self, video_name, images_path, enforce_iframes,
                              im_width=640, im_height=480):
        final_results = self.app.create_empty_results()
        number_of_frames = len(
            [e for e in os.listdir(images_path) if "png" in e])
        last_sent = None
        tracking_success = True
        total_sent = 0
        last_sent_id = None

        # initialize server
        self.init_server()

        for fid in range(0, number_of_frames):
            fname = f"{str(fid).zfill(10)}.png"
            frame_path = os.path.join(images_path, fname)
            curr_frame = cv.imread(frame_path)
            curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)

            if (self.is_trigger_frame_glimpse(
                    last_sent, curr_gray) or not tracking_success) and (
                        not last_sent_id or (fid - last_sent_id > 5)):

                self.logger.info(f"Sending frame {fid} to server")
                req_regions = Regions()
                req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                        self.config.low_resolution))
                batch_video_size, _ = compute_regions_size(
                    req_regions, f"{video_name}-base-phase", images_path,
                    self.config.low_resolution, self.config.low_qp,
                    enforce_iframes, True)
                self.logger.info(f"{batch_video_size // 1024}KB sent "
                            f"in base phase using {self.config.low_qp}QP")
                results, _ = self.get_first_phase_results(video_name, fid, fid + 1)
                #for r in detection_results.regions:
                #    r.origin = "glimpse-detection"
                last_sent = curr_gray
                final_results.combine_results(results, self.config)
                last_sent_id = fid
                tracking_success = True
                self.initialize_trackers(results, fid, images_path)

                #total_sent += os.path.getsize(f"{video_name}_temp.jpg")
                #os.remove(f"{video_name}_temp.jpg")

            # Pick objects in last frame and perform tracking in current frame
            if fid > 0 and fid != last_sent_id:
                tracking_results, tracking_success = self.track(
                    final_results, fid, images_path)
                self.logger.info(f"Got {len(tracking_results)} from tracking frame {fid}")
                #for r in tracking_results.regions:
                #    r.origin = "glimpse-tracking"
                final_results.combine_results(tracking_results, self.config)

        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        final_results.write(video_name)

        return final_results, [total_sent, 0]


    def analyze_video_vigil_emulate(self, video_name, images_path):
        local_detector = LocalObjectDetector(
            "../MobileNetSSD.pb", "../ssd_graph.pbtxt", 0.5)
        number_of_frames = len(
            [e for e in os.listdir(images_path) if "png" in e])
        last_count = None
        final_results = Regions()
        tracking_success = False
        total_size = 0
        last_sent_id = None

        for fid in range(0, number_of_frames):
            fname = f"{str(fid).zfill(10)}.png"
            frame_path = os.path.join(images_path, fname)
            curr_frame = cv.imread(frame_path)
            _, num_objects_in_frame = local_detector.infer(curr_frame)
            if (self.is_important_frame(
                    last_count, num_objects_in_frame) or not tracking_success) and (
                        not last_sent_id or (fid - last_sent_id > 5)):
                cv.imwrite(f"{video_name}_temp.jpg", curr_frame,
                           [cv.IMWRITE_JPEG_QUALITY, 70])
                reread_current_frame = cv.imread(f"{video_name}_temp.jpg")

                # run inference
                results = self.app.run_inference(
                    self.server.model, None, 1.0, [fname], {fid: reread_current_frame}, config=self.config)
                detection_results = results["results"]

                for fid in detection_results.regions_dict:
                    for r in detection_results.regions_dict[fid]:
                        r.origin = "vigil-detection"
                final_results.combine_results(detection_results, 1.0)
                last_sent_id = fid
                tracking_success = True
                # WARNING: same issue as in glimpse
                #self.initialize_trackers(detection_results, fid, images_path)
                self.initialize_trackers(final_results, fid, images_path)

                total_size += os.path.getsize(f"{video_name}_temp.jpg")
                os.remove(f"{video_name}_temp.jpg")
            last_count = num_objects_in_frame

            if fid > 0 and fid != last_sent_id:
                tracking_results, tracking_success = self.track(
                    final_results, fid, images_path)
                self.logger.info(f"Got {len(tracking_results)} from tracking frame {fid}")
                for fid in tracking_results.regions_dict:
                    for r in tracking_results.regions_dict[fid]:
                        r.origin = "tracking-vigil"
                final_results.combine_results(tracking_results, 1.0)

        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        final_results.write(video_name)

        return final_results, [total_size, 0]
    

    def analyze_video_vigil(self, video_name, images_path, enforce_iframes):
        local_detector = LocalObjectDetector(
            "../MobileNetSSD.pb", "../ssd_graph.pbtxt", 0.5)
        number_of_frames = len(
            [e for e in os.listdir(images_path) if "png" in e])
        last_count = None
        final_results = Regions()
        tracking_success = False
        total_size = 0
        last_sent_id = None

        # initialize server
        self.init_server()

        for fid in range(0, number_of_frames):
            fname = f"{str(fid).zfill(10)}.png"
            frame_path = os.path.join(images_path, fname)
            curr_frame = cv.imread(frame_path)
            _, num_objects_in_frame = local_detector.infer(curr_frame)

            if (self.is_important_frame(
                    last_count, num_objects_in_frame) or not tracking_success) and (
                        not last_sent_id or (fid - last_sent_id > 5)):

                self.logger.info(f"Sending frame {fid} to server")
                req_regions = Regions()
                req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                        self.config.low_resolution))
                batch_video_size, _ = compute_regions_size(
                    req_regions, f"{video_name}-base-phase", images_path,
                    self.config.low_resolution, self.config.low_qp,
                    enforce_iframes, True)
                self.logger.info(f"{batch_video_size // 1024}KB sent "
                            f"in base phase using {self.config.low_qp}QP")
                detection_results, _ = self.get_first_phase_results(video_name, fid, fid + 1)

                for fid in detection_results.regions_dict:
                    for r in detection_results.regions_dict[fid]:
                        r.origin = "vigil-detection"
                final_results.combine_results(detection_results, 1.0)
                last_sent_id = fid
                tracking_success = True
                # WARNING: same issue as in glimpse
                #self.initialize_trackers(detection_results, fid, images_path)
                self.initialize_trackers(final_results, fid, images_path)

                # WARNING: no bandwidth estimation for now
                #total_size += os.path.getsize(f"{video_name}_temp.jpg")
                #os.remove(f"{video_name}_temp.jpg")
            last_count = num_objects_in_frame

            if fid > 0 and fid != last_sent_id:
                tracking_results, tracking_success = self.track(
                    final_results, fid, images_path)
                self.logger.info(f"Got {len(tracking_results)} from tracking frame {fid}")
                for fid in tracking_results.regions_dict:
                    for r in tracking_results.regions_dict[fid]:
                        r.origin = "tracking-vigil"
                final_results.combine_results(tracking_results, 1.0)

        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        final_results.write(video_name)

        return final_results, [total_size, 0]
    

    def is_important_frame(self, last_count, curr_count):
        if not last_count or abs(curr_count - last_count):
            return True
        else:
            return False