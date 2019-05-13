import os
import logging
import cv2 as cv
from dds_utils import ServerConfig, Results, Region


class Server:
    """The server component of DDS protocol. Responsible for running DNN
       on low resolution images, tracking to find regions of interest and
       running DNN on the high resolution regions of interest"""

    def __init__(self, high_threshold, low_threshold,
                 max_object_size, tracker_length, boundary):
        self.conf = ServerConfig(high_threshold, low_threshold,
                                 max_object_size, tracker_length, boundary)

        self.logger = logging.getLogger("server")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.logger.info("Server started")

    def track(self, obj_to_track, start_fid, end_fid, images_direc):
        regions = Results()

        # Extract extra object properties to add to region
        conf = obj_to_track.conf
        label = obj_to_track.label
        resolution = obj_to_track.resolution

        init_fname = os.path.join(images_direc,
                                  f"{str(start_fid).zfill(10)}.png")
        init_frame = cv.imread(init_fname)
        im_width = init_frame.shape[1]
        im_height = init_frame.shape[0]

        # Initialize Tracker
        x = obj_to_track.x * im_width
        y = obj_to_track.y * im_height
        w = obj_to_track.w * im_width
        h = obj_to_track.h * im_height
        bbox = (x, y, w, h)
        tracker = cv.TrackerKCF_create()
        tracker.init(init_frame, bbox)

        # Make the frame range
        frame_range = None
        if start_fid < end_fid:
            frame_range = range(start_fid, end_fid)
        else:
            frame_range = range(start_fid, end_fid, -1)
        # Remove the first frame which was used to initialize the tracker
        frame_range = frame_range[1:]
        for fid in frame_range:
            curr_frame_fname = f"{str(fid).zfill(10)}.png"
            curr_frame_path = os.path.join(images_direc, curr_frame_fname)
            curr_frame = cv.imread(curr_frame_path)

            status, bbox = tracker.update(curr_frame)
            if status:
                # Add bounding box to areas to be searched
                x = bbox[0] / im_width
                y = bbox[1] / im_height
                w = bbox[2] / im_width
                h = bbox[3] / im_height
                region = Region(fid, x, y, w, h, conf, label, resolution)
                regions.add_single_result(region)

        return regions

    def get_regions_to_query(self, start_fid, end_fid, images_direc, results,
                             config, simulation=False):
        non_tracking_regions = Results()
        tracking_regions = Results()

        for single_result in results.regions:
            if (single_result.conf > config.low_threshold and
                    single_result.conf < config.high_threshold):
                # These are only those regions which are between thresholds
                non_tracking_regions.add_single_result(single_result)

            # Even if the results is not between thresholds we still need to
            # Track it across frames
            start_frame = single_result.fid
            self.logger.debug(f"Finding regions to query for "
                              f"{single_result.to_str()}")

            # Forward tracking
            end_frame = min(start_frame + config.tracker_length, end_fid)
            regions_from_tracking = self.track(single_result, start_frame,
                                               end_frame, images_direc)
            self.logger.debug(f"Found {regions_from_tracking.results_len()} "
                              f"regions using forward tracking from"
                              f" {start_frame} to {end_frame}")
            tracking_regions.combine_results(regions_from_tracking)

            # Backward tracking
            end_frame = max(0, start_frame - config.tracker_length)
            regions_from_tracking = self.track(single_result, start_frame,
                                               end_frame, images_direc)
            self.logger.debug(f"Found {regions_from_tracking.results_len()} "
                              f"regions using backward tracking from"
                              f" {start_frame} to {end_frame}")
            tracking_regions.combine_results(regions_from_tracking)

        self.logger.info(f"Found {non_tracking_regions.results_len()} "
                         f"regions between {start_fid} and {end_fid} without "
                         f"tracking")
        self.logger.info(f"Found {tracking_regions.results_len()} regions "
                         f"between {start_fid} and {end_fid} with tracking")

        # Enlarge regions iff we are running a simulation
        if not simulation:
            # Enlarge non-tracking boxes
            for result in non_tracking_regions.regions:
                new_x = max(result.x - config.boundary * result.w, 0)
                new_y = max(result.y - config.boundary * result.h, 0)
                new_w = min(result.w + config.boundary * result.w * 2,
                            1 - result.x + config.boundary * result.w)
                new_h = min(result.h + config.boundary * result.h * 2,
                            1 - result.y + config.boundary * result.h)

                result.x = new_x
                result.y = new_y
                result.w = new_w
                result.h = new_h

            # Enlarge tracking boxes
            for result in tracking_regions.regions:
                new_x = max(result.x - 2 * config.boundary * result.w, 0)
                new_y = max(result.y - 2 * config.boundary * result.h, 0)
                new_w = min(result.w + 2 * config.boundary * result.w * 2,
                            1 - result.w + 2 * config.boundary * result.w)
                new_h = min(result.h + 2 * config.boundary * result.h * 2,
                            1 - result.h + 2 * config.boundary * result.h)

                result.x = new_x
                result.y = new_y
                result.w = new_w
                result.h = new_h

        final_regions = Results()
        final_regions.combine_results(non_tracking_regions)
        final_regions.combine_results(tracking_regions)

        return final_regions

    def simulate_low_query(self, start_fid, end_fid, images_direc,
                           results_dict, config=None):
        curr_conf = self.conf
        if config is not None:
            curr_conf = config

        results = Results()
        accepted_results = Results()
        results_for_regions = Results()  # Results used for regions detection

        # Extract relevant results
        for fid in range(start_fid, end_fid):
            fid_results = results_dict[fid]
            for single_result in fid_results:
                results.add_single_result(single_result)

        self.logger.info(f"Getting results with threshold "
                         f"{curr_conf.low_threshold} and "
                         f"{curr_conf.high_threshold}")

        for single_result in results.regions:
            if single_result.conf < curr_conf.low_threshold:
                continue

            results_for_regions.add_single_result(single_result)

            if single_result.conf > config.high_threshold:
                accepted_results.add_single_result(single_result)

        regions_to_query = self.get_regions_to_query(start_fid, end_fid,
                                                     images_direc,
                                                     results_for_regions,
                                                     curr_conf,
                                                     simulation=True)
        self.logger.info(f"Returning {accepted_results.results_len()} "
                         f"confirmed results and "
                         f"{regions_to_query.results_len()} regions")

        # Return results and regions
        return accepted_results, regions_to_query

    def simulate_high_query(self, req_regions, high_results_dict, config=None):
        curr_conf = self.conf
        if config is not None:
            curr_conf = config

        high_res_results = Results()

        # Get all results that have confidence above the threshold and
        # is in one of the frames in the queried regions
        fids_in_queried_regions = [e.fid for e in req_regions.regions]
        for fid in fids_in_queried_regions:
            fid_results = high_results_dict[fid]
            for single_result in fid_results:
                confidence = single_result.conf
                if confidence > curr_conf.high_threshold:
                    high_res_results.add_single_result(single_result)

        selected_results = Results()
        for single_result in high_res_results.regions:
            if req_regions.is_dup(single_result):
                self.logger.debug(f"Matched {single_result.to_str()} in "
                                  f"requested regions")
                selected_results.add_single_result(single_result)

        return selected_results
