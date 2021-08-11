import logging
import os
import shutil
import cv2 as cv
import numpy as np
import time
from .object_detector import LocalObjectDetector
from dds_utils import (Results, read_results_dict, cleanup, Region,
                       compute_regions_size, extract_images_from_video,
                       merge_boxes_in_results, calc_frame_difference, visualize_regions)


class Client:
    """The client of the DDS protocol
       sends images in low resolution and waits for
       further instructions from the server. And finally receives results
       Note: All frame ranges are half open ranges"""

    def __init__(self, server_handle, hname, config):
        self.hname = hname
        self.server = server_handle
        self.config = config

        self.logger = logging.getLogger("client")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.trackers = {}

        self.logger.info(f"Client initialized")

    def analyze_video_mpeg(self, video_name, raw_images_path, enforce_iframes):

        number_of_frames = len(
            [f for f in os.listdir(raw_images_path) if ".png" in f])

        final_results = Results()
        final_rpn_results = Results()
        total_size = 0

        # Kuntai: for delay logging
        segment_size_file = open(f'{video_name}_segment_size', 'w')
        import timeit

        for i in range(0, number_of_frames, self.config.batch_size):

            # Kuntai: add timer for delay logging
            start = timeit.default_timer()

            start_frame = i
            end_frame = min(number_of_frames, i + self.config.batch_size)

            batch_fnames = sorted([f"{str(idx).zfill(10)}.png"
                                   for idx in range(start_frame, end_frame)])

            req_regions = Results()
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
            results, rpn_results = (
                self.server.perform_detection(
                    f"{video_name}-base-phase-cropped",
                    self.config.low_resolution, batch_fnames))

            self.logger.info(f"Detection {len(results)} regions for "
                             f"batch {start_frame} to {end_frame} with a "
                             f"total size of {batch_video_size / 1024}KB")
            final_results.combine_results(
                results, self.config.intersection_threshold)
            final_rpn_results.combine_results(
                rpn_results, self.config.intersection_threshold)

            # Remove encoded video manually
            shutil.rmtree(f"{video_name}-base-phase-cropped")
            total_size += batch_video_size

            # Kuntai: check the time of that batch
            stop = timeit.default_timer()
            batch_time = stop - start
            segment_size_file.write(f"{end_frame-start_frame+1}, {batch_time}, {batch_video_size}\n")

        # Kuntai: close the segment file.
        segment_size_file.close()

        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        final_results.fill_gaps(number_of_frames)

        # Add RPN regions
        final_results.combine_results(
            final_rpn_results, self.config.intersection_threshold)

        final_results.write(video_name)

        return final_results, [total_size, 0]

    def analyze_video_emulate(self, video_name, high_images_path,
                              enforce_iframes, low_results_path=None,
                              debug_mode=False):
        segment_record_file = open(f"{video_name}--sizes", "w")
        current_schedule = int(os.environ["CURR_SCHEDULE"])
        config_to_high_map = {}
        with open("dds.knobtable") as f:
            lines = [l.split(",") for l in f.readlines()[1:]]
            for line in lines:
                config_to_high_map[int(line[0])] = int(line[1])
        high_to_res_qp_map = {}
        with open("highConfiguration") as f:
            lines = [l.split(" ") for l in f.readlines()]
            for line in lines:
                high_to_res_qp_map[int(line[0])] = (float(line[1]), int(line[2]))
        batch_to_res_qp_map = {}
        with open(f"schedule-{current_schedule}") as f:
            lines = f.readlines()
            for line in lines:
                splitted = line.split(" ")
                schedule_idx = int(splitted[1][:-1])
                config_idx = int(splitted[4])
                batch_to_res_qp_map[schedule_idx] = high_to_res_qp_map[
                    config_to_high_map[config_idx]]

        final_results = Results()
        low_phase_results = Results()
        high_phase_results = Results()

        number_of_frames = len(
            [x for x in os.listdir(high_images_path) if "png" in x])

        low_results_dict = None
        if low_results_path:
            low_results_dict = read_results_dict(low_results_path)

        total_size = [0, 0]
        total_regions_count = 0
        for idx, i in enumerate(range(0, number_of_frames,
                                      self.config.batch_size)):
            self.config.high_resolution, self.config.high_qp = (
                batch_to_res_qp_map[idx])

            start_fid = i
            end_fid = min(number_of_frames, i + self.config.batch_size)
            self.logger.info(f"Processing batch from {start_fid} to {end_fid}")

            low_size, high_size = 0, 0
            # Encode frames in batch and get size
            # Make temporary frames to downsize complete frames
            base_req_regions = Results()
            for fid in range(start_fid, end_fid):
                base_req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.high_resolution))
            encoded_batch_video_size, batch_pixel_size = compute_regions_size(
                base_req_regions, f"{video_name}-base-phase", high_images_path,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"Sent {encoded_batch_video_size / 1024} "
                             f"in base phase")
            low_size = encoded_batch_video_size
            total_size[0] += encoded_batch_video_size

            # Low resolution phase
            low_images_path = f"{video_name}-base-phase-cropped"
            r1, req_regions = self.server.simulate_low_query(
                start_fid, end_fid, low_images_path, low_results_dict, False,
                self.config.rpn_enlarge_ratio)
            total_regions_count += len(req_regions)

            low_phase_results.combine_results(
                r1, self.config.intersection_threshold)
            final_results.combine_results(
                r1, self.config.intersection_threshold)

            # High resolution phase
            if len(req_regions) > 0:
                # Crop, compress and get size
                regions_size, _ = compute_regions_size(
                    req_regions, video_name, high_images_path,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                self.logger.info(f"Sent {len(req_regions)} regions which have "
                                 f"{regions_size / 1024}KB in second phase "
                                 f"using {self.config.high_qp}")
                high_size = regions_size
                total_size[1] += regions_size

                # High resolution phase every three filter
                r2 = self.server.emulate_high_query(
                    video_name, low_images_path, req_regions)
                self.logger.info(f"Got {len(r2)} results in second phase "
                                 f"of batch")

                high_phase_results.combine_results(
                    r2, self.config.intersection_threshold)
                final_results.combine_results(
                    r2, self.config.intersection_threshold)

            segment_record_file.write(
                f"{idx} {low_size} {high_size} "
                f"{self.config.low_resolution} {self.config.low_qp} "
                f"{self.config.high_resolution} {self.config.low_qp}\n")
            # Cleanup for the next batch
            cleanup(video_name, debug_mode, start_fid, end_fid)

        self.logger.info(f"Got {len(low_phase_results)} unique results "
                         f"in base phase")
        self.logger.info(f"Got {len(high_phase_results)} positive "
                         f"identifications out of {total_regions_count} "
                         f"requests in second phase")

        # Fill gaps in results
        final_results.fill_gaps(number_of_frames)

        # Write results
        final_results.write(f"{video_name}")

        self.logger.info(f"Writing results for {video_name}")
        self.logger.info(f"{len(final_results)} objects detected "
                         f"and {total_size[1]} total size "
                         f"of regions sent in high resolution")

        rdict = read_results_dict(f"{video_name}")
        final_results = merge_boxes_in_results(rdict, 0.3, 0.3)

        final_results.fill_gaps(number_of_frames)
        final_results.write(f"{video_name}")
        return final_results, total_size

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
        tracking_results = Results()
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

    def analyze_video_vigil(self, video_name, images_path):
        local_detector = LocalObjectDetector(
            "MobileNetSSD.pb", "ssd_graph.pbtxt", 0.5)
        number_of_frames = len(
            [e for e in os.listdir(images_path) if "png" in e])
        last_count = None
        final_results = Results()
        tracking_success = False
        total_size = 0
        sent_frames=0
        last_sent_id = None

        # Kuntai: for logging the time!
        import timeit
        segment_size_file = open(f'{video_name}_segment_size', 'w')
        batch_video_size = 0


        for fid in range(0, number_of_frames):
            # Kuntai: log the start time
            start = timeit.default_timer()

            fname = f"{str(fid).zfill(10)}.png"
            frame_path = os.path.join(images_path, fname)
            curr_frame = cv.imread(frame_path)
            bboxes, num_objects_in_frame = local_detector.infer(curr_frame)

            if (self.is_important_frame(
                    last_count, num_objects_in_frame) or not tracking_success) and (
                        not last_sent_id or (fid - last_sent_id > 5)):


                # Kuntai: black out background
                print(fid)
                sent_frames+=1
                curr_frame_black_background = curr_frame.copy()
                curr_frame_black_background[:, :, :] = 0
                im_height, im_width, _ = curr_frame.shape
                for bbox in bboxes[0, 0]:
                    # adding this will miss TONS OF cars!
                    # if int(bbox[1]) not in [6, 7]:
                    #    continue
                    if float(bbox[2]) > local_detector.confidence_threshold:
                        continue
                    bbox = bbox[3:7] * np.array([im_width, im_height, im_width, im_height])
                    bbox[2] = max(bbox[2], bbox[0])
                    bbox[3] = max(bbox[3], bbox[1])
                    for i in range(4):
                        bbox[i] = max(min(bbox[i], [im_width, im_height, im_width, im_height][i] - 1), 0)
                    bbox = bbox.astype(int)
                    curr_frame_black_background[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1, :] = curr_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1, :]
                cv.imwrite(f"temp_{fid}.jpg", curr_frame_black_background,
                           [cv.IMWRITE_JPEG_QUALITY, 70])


                cv.imwrite(f"{video_name}_temp.jpg", curr_frame,
                           [cv.IMWRITE_JPEG_QUALITY, 70])
                reread_current_frame = cv.imread(f"{video_name}_temp.jpg")
                detection_results, _ = self.server.perform_detection(
                    None, 1.0, [fname], {fid: reread_current_frame})
                for r in detection_results.regions:
                    r.origin = "vigil-detection"
                final_results.combine_results(detection_results, 1.0)
                last_sent_id = fid
                tracking_success = True
                self.initialize_trackers(detection_results, fid, images_path)

                total_size += os.path.getsize(f"temp_{fid}.jpg")

                # Kuntai: memorize the size info
                batch_video_size = os.path.getsize(f"temp_{fid}.jpg")

                os.remove(f"{video_name}_temp.jpg")
            last_count = num_objects_in_frame

            if fid > 0 and fid != last_sent_id:
                tracking_results, tracking_success = self.track(
                    final_results, fid, images_path)
                self.logger.info(f"Got {len(tracking_results)} from tracking frame {fid}")
                for r in tracking_results.regions:
                    r.origin = "tracking-vigil"
                final_results.combine_results(tracking_results, 1.0)


            # Kuntai: log the end time
            stop = timeit.default_timer()
            batch_time = stop - start
            segment_size_file.write(f"1, {batch_time}, {batch_video_size}\n")

        # Kuntai: close the time logging file
        #segment_size_file.close()
        with open('vigil.txt', 'a') as f:
            f.write(f'{sent_frames/number_of_frames}\n')

        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        final_results.write(video_name)

        return final_results, [total_size, 0]



    def analyze_video_glimpse(self, video_name, images_path,
                              im_width=640, im_height=480):
        number_of_frames = len(
            [e for e in os.listdir(images_path) if "png" in e])
        last_sent = None
        final_results = Results()
        tracking_success = True
        total_sent = 0
        sent_frames = 0
        last_sent_id = None

        # Kuntai: Still, for logging the time...
        import timeit
        segment_size_file = open(f'{video_name}_segment_size', 'w')


        for fid in range(0, number_of_frames):

            # Kuntai: log the start time
            start = timeit.default_timer()
            batch_video_size = 0


            fname = f"{str(fid).zfill(10)}.png"
            frame_path = os.path.join(images_path, fname)
            curr_frame = cv.imread(frame_path)
            curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
            if (self.is_trigger_frame_glimpse(
                    last_sent, curr_gray) or not tracking_success) and (
                        not last_sent_id or (fid - last_sent_id > 5)):
                print(fid)
                sent_frames+=1
                resized_frame = cv.resize(curr_frame, (im_height, im_width))
                self.logger.info(f"Sending frame to server {fid}")
                cv.imwrite(f"{video_name}_temp.jpg", resized_frame,
                           [cv.IMWRITE_JPEG_QUALITY, 70])
                resized_frame = cv.imread(f"{video_name}_temp.jpg")
                detection_results, _ = self.server.perform_detection(
                    None, 1.0, [fname], {fid: resized_frame})
                for r in detection_results.regions:
                    r.origin = "glimpse-detection"
                last_sent = curr_gray
                final_results.combine_results(detection_results, 1.0)
                last_sent_id = fid
                tracking_success = True
                self.initialize_trackers(detection_results, fid, images_path)

                total_sent += os.path.getsize(f"{video_name}_temp.jpg")

                # Kuntai: memorize the size info
                batch_video_size = os.path.getsize(f"{video_name}_temp.jpg")

                os.remove(f"{video_name}_temp.jpg")

            # Pick objects in last frame and perform tracking in current frame
            if fid > 0 and fid != last_sent_id:
                tracking_results, tracking_success = self.track(
                    final_results, fid, images_path)
                self.logger.info(f"Got {len(tracking_results)} from tracking frame {fid}")
                for r in tracking_results.regions:
                    r.origin = "glimpse-tracking"
                final_results.combine_results(tracking_results, 1.0)

            # Kuntai: log the end time
            stop = timeit.default_timer()
            batch_time = stop-start
            segment_size_file.write(f"1, {batch_time}, {batch_video_size}\n")

        # Kuntai: close the logging file
        #segment_size.close()

        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        final_results.write(video_name)

        with open('glimpse.txt', 'a') as f:
            f.write(f'{sent_frames/number_of_frames}\n')

        return final_results, [total_sent, 0]

    def is_important_frame(self, last_count, curr_count):
        if not last_count or abs(curr_count - last_count):
            return True
        else:
            return False
