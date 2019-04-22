import os
import cv2 as cv
from ..utils import ServerConfig, Results, Region


class Server:
    def __init__(self, high_threshold, low_threshold,
                 max_object_size, tracker_length):
        self.conf = ServerConfig(high_threshold, low_threshold,
                                 max_object_size, tracker_length)

    def track(self, obj_to_track, start_fid, end_fid, images_direc):
        regions = Results()

        # Extract extra object properties to add to region
        conf = obj_to_track.conf
        label = obj_to_track.label
        resolution = obj_to_track.resolution

        init_fname = os.path.join(images_direc,
                                  "{}.png".format(str(start_fid).zfill(10)))
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

        # Remove first frame, used for initializing the tracker
        frame_range = range(start_fid, end_fid)[1:]
        for fid in frame_range:
            curr_frame_fname = "{}.png".format(str(fid).zfill(10))
            curr_frame_path = os.paht.join(images_direc, curr_frame_fname)
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

    def simulate_low_query(self, start_fid, end_fid, images_direc,
                           results_dict, config=None):
        curr_conf = self.conf
        if config is not None:
            curr_conf = config

        results = Results()
        accepted_results = Results()
        regions_to_query = Results()

        # Extract relevant results
        for fid in range(start_fid, end_fid):
            fid_results = results_dict[fid]
            for single_result in fid_results:
                confidence = single_result.conf
                if confidence > curr_conf.low_threshold:
                    results.add_single_result(single_result)

        # Tracking phase
        for single_result in results.single_obj_results:
            # Accept result as accepted if confidence greater than a threshold
            if single_result.conf > curr_conf.high_threshold:
                accepted_results.add_single_result(single_result)

            start_frame = single_result.fid

            # Forward tracking
            end_frame = min(start_frame + curr_conf.tracker_length, end_fid)
            regions_from_tracking = self.track(single_result, start_frame,
                                               end_frame, images_direc)
            regions_to_query.combine_results(regions_to_query)

            # Backward tracking
            end_frame = max(0, start_frame - curr_conf.tracker_length)
            regions_from_tracking = self.track(single_result, start_frame,
                                               end_frame, images_direc)
            regions_to_query.combine_results(regions_from_tracking)

        # Remove regions that are in the accepted results
        final_regions_to_query = Results()
        for region in regions_to_query.single_obj_results:
            if not accepted_results.is_dup(region):
                final_regions_to_query.add_single_result(region)

        # Return results and regions
        return results, final_regions_to_query
