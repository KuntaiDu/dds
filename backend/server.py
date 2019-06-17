import os
import logging
import cv2 as cv
from dds_utils import (Results, Region, calc_intersection_area,
                       calc_area)
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

    def perform_detection(self, images_direc, resolution, fnames=None):
        final_results = Results()
        if fnames is None:
            fnames = sorted(os.listdir(images_direc))

        for fname in fnames:
            if "png" not in fname:
                continue
            image_path = os.path.join(images_direc, fname)
            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            detection_results = self.detector.infer(image)

            fid = int(fname.split(".")[0])

            if not detection_results:
                r = Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution)
                final_results.append(r)

            for label, conf, (x, y, w, h) in detection_results:
                r = Region(fid, x, y, w, h, conf, label,
                           resolution, origin="mpeg")
                final_results.append(r)

        return final_results

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
                x = max(0, bbox[0] / im_width)
                y = max(0, bbox[1] / im_height)
                w = bbox[2] / im_width
                h = bbox[3] / im_height

                region = Region(fid, x, y, w, h, conf, label, resolution,
                                f"tracking-extension[{start_fid}-{end_fid}]")
                regions.append(region)
            else:
                break

        return regions

    def get_regions_to_query(self, start_fid, end_fid, images_direc,
                             results_for_tracking, accepted_results,
                             simulation=False):
        non_tracking_regions = Results()
        tracking_regions = Results()

        for single_result in results_for_tracking.regions:
            single_result = single_result.copy()

            if (single_result.conf < self.config.low_threshold or
                    ((single_result.w * single_result.h) >
                     self.config.max_object_size)):
                # Don't investivagte if the size is too large
                # or if confidence is below low threshold
                continue

            if (single_result.conf < self.config.high_threshold):
                # These are only those regions which are between thresholds
                single_result.origin = f"tracking-origin[{single_result.fid}]"
                non_tracking_regions.add_single_result(
                    single_result, self.config.intersection_threshold)

            # Even if the results is not between thresholds we still need to
            # Track it across frames
            start_frame = single_result.fid
            self.logger.debug(f"Finding regions to query for "
                              f"{single_result}")

            # Forward tracking
            end_frame = min(start_frame + self.config.tracker_length, end_fid)
            regions_from_tracking = self.track(single_result, start_frame,
                                               end_frame, images_direc)
            self.logger.debug(f"Found {len(regions_from_tracking)} "
                              f"regions using forward tracking from"
                              f" {start_frame} to {end_frame}")
            tracking_regions.combine_results(
                regions_from_tracking, self.config.intersection_threshold)

            # Backward tracking
            end_frame = max(start_fid,
                            start_frame - self.config.tracker_length)

            regions_from_tracking = self.track(single_result, start_frame,
                                               end_frame, images_direc)
            self.logger.debug(f"Found {len(regions_from_tracking)} "
                              f"regions using backward tracking from"
                              f" {start_frame} to {end_frame}")
            tracking_regions.combine_results(
                regions_from_tracking, self.config.intersection_threshold)

        self.logger.info(f"Found {len(non_tracking_regions)} "
                         f"regions between {start_fid} and {end_fid} without "
                         f"tracking")
        self.logger.info(f"Found {len(tracking_regions)} regions "
                         f"between {start_fid} and {end_fid} with tracking")

        # Enlarge regions iff we are running a simulation
        if not simulation:
            # Enlarge non-tracking boxes
            for result in non_tracking_regions.regions:
                new_x = max(result.x - self.config.boundary * result.w, 0)
                new_y = max(result.y - self.config.boundary * result.h, 0)
                new_w = min(result.w + self.config.boundary * result.w * 2,
                            1 - result.x + self.config.boundary * result.w)
                new_h = min(result.h + self.config.boundary * result.h * 2,
                            1 - result.y + self.config.boundary * result.h)

                result.x = new_x
                result.y = new_y
                result.w = new_w
                result.h = new_h

            # Enlarge tracking boxes
            for result in tracking_regions.regions:
                new_x = max(result.x - 2 * self.config.boundary * result.w, 0)
                new_y = max(result.y - 2 * self.config.boundary * result.h, 0)
                new_w = min(result.w + 2 * self.config.boundary * result.w * 2,
                            1 - result.w + 2 * self.config.boundary * result.w)
                new_h = min(result.h + 2 * self.config.boundary * result.h * 2,
                            1 - result.h + 2 * self.config.boundary * result.h)

                result.x = new_x
                result.y = new_y
                result.w = new_w
                result.h = new_h

        final_regions = Results()
        # Add all non_tracking_regions
        final_regions.combine_results(
            non_tracking_regions, self.config.intersection_threshold)
        # Cleanup tracking regions
        for region in tracking_regions.regions:
            if region.w * region.h > self.config.max_object_size:
                # Skip if size too large
                continue
            matched_region = accepted_results.is_dup(region, 0.3)
            if not matched_region:
                final_regions.append(region)
            elif matched_region.conf < self.config.high_threshold:
                final_regions.append(region)
            final_regions.label = "-1"

        return final_regions

    def simulate_low_query(self, start_fid, end_fid, images_direc,
                           results_dict):
        results = Results()
        accepted_results = Results()
        results_for_regions = Results()  # Results used for regions detection

        # Extract relevant results
        for fid in range(start_fid, end_fid):
            fid_results = results_dict[fid]
            for single_result in fid_results:
                single_result.origin = "low-res"
                results.add_single_result(single_result,
                                          self.config.intersection_threshold)

        self.logger.info(f"Getting results with threshold "
                         f"{self.config.low_threshold} and "
                         f"{self.config.high_threshold}")

        for single_result in results.regions:
            accepted_results.add_single_result(
                single_result, self.config.intersection_threshold)

            if single_result.conf < self.config.low_threshold:
                continue

            results_for_regions.add_single_result(
                single_result, self.config.intersection_threshold)

        regions_to_query = self.get_regions_to_query(start_fid, end_fid,
                                                     images_direc,
                                                     results_for_regions,
                                                     accepted_results,
                                                     simulation=True)
        self.logger.info(f"Returning {len(accepted_results)} "
                         f"confirmed results and "
                         f"{len(regions_to_query)} regions")

        # Return results and regions
        return accepted_results, regions_to_query

    def simulate_high_query(self, req_regions, high_results_dict):
        high_res_results = Results()

        # Get all results that have confidence above the threshold and
        # is in one of the frames in the queried regions
        fids_in_queried_regions = [e.fid for e in req_regions.regions]
        for fid in fids_in_queried_regions:
            fid_results = high_results_dict[fid]
            for single_result in fid_results:
                single_result.origin = "high-res"
                high_res_results.add_single_result(
                    single_result, self.config.intersection_threshold)

        # Iterate over high_res_results to ensure that all matching regions
        # are added. Iterating over required regions would just add one
        # high resolution region for a requested region
        high_res_regions_to_del = []
        selected_results = Results()
        for single_result in high_res_results.regions:
            dup_region = req_regions.is_dup(
                single_result, self.config.intersection_threshold)
            if dup_region:
                self.logger.debug(f"Matched {single_result} with "
                                  f"{dup_region} "
                                  f"in requested regions from IOU")
                single_result.origin = "high-res"
                selected_results.add_single_result(
                    single_result, self.config.intersection_threshold)
                high_res_regions_to_del.append(single_result)
        # Delete the high resolution regions that have already been added to
        # selected_results
        for region in high_res_regions_to_del:
            high_res_results.remove(region)

        # Add regions based on intersection alone
        for high_region in high_res_results.regions:
            for req_region in req_regions.regions:
                intersection = calc_intersection_area(high_region, req_region)
                if intersection > 0.8 * calc_area(high_region):
                    self.logger.debug(f"Matched {high_region} with "
                                      f"{req_region} "
                                      f"in requested regions from "
                                      f"intersection")
                    high_region.origin = "high-res"
                    selected_results.add_single_result(
                        high_region, self.config.intersection_threshold)

        return selected_results

    def emulate_low_query(self, start_fid, end_fid, low_images_path):
        batch_fnames = sorted([f"str(i).zfill(10).png"
                               for i in range(start_fid, end_fid)])
        detection_results = self.perform_detection(
            low_images_path, self.config.low_resolution, batch_fnames)

        results_dict = {}
        for region in detection_results.regions:
            # Do not add no obj detections in the dictionary
            if region.label == "no obj":
                continue
            if region.fid not in results_dict:
                results_dict[region.fid] = []
            results_dict[region.fid].append(region)

        accepted_results, final_regions_to_query = self.simulate_low_query(
            start_fid, end_fid, low_images_path, results_dict)

        return accepted_results, final_regions_to_query

    def emulate_high_query(self, req_regions, images_direc):
        req_fids = list(set([r.fid for r in req_regions.regions]))
        req_fids = sorted(req_fids)

        fnames = sorted([f for f in os.listdir(images_direc) if "png" in f])

        results = self.perform_detection(images_direc,
                                         self.config.high_resolution, fnames)

        # Remap fid in results to fid in req regions
        results_fids = list(set([r.fid for r in results.regions]))
        fid_to_fid_mapping = {}
        for idx in range(len(results_fids)):
            fid_to_fid_mapping[results_fids[idx]] = req_fids[idx]
        for r in results.regions:
            r.fid = fid_to_fid_mapping[r.fid]

        results_with_detections_only = Results()
        for r in results.regions:
            if r.label == "no obj":
                continue
            results_with_detections_only.append(r)

        return results_with_detections_only
