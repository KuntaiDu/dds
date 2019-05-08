import logging
import os
from dds_utils import (Results, read_results_dict, write_results,
                       ServerConfig, compute_area_of_regions)


class Client:
    """The client of the DDS protocol
       sends images in low resolution and waits for
       further instructions from the server. And finally receives results"""

    def __init__(self, server_handle, hname=None, high_threshold=0.8,
                 low_threshold=0.3, max_object_size=0.3,
                 tracker_length=4, boundary=0.2):
        self.hname = hname
        self.server = server_handle
        self.server_conf = ServerConfig(high_threshold, low_threshold,
                                        max_object_size,
                                        tracker_length, boundary)

        self.logger = logging.getLogger("client")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        if hname is None:
            self.logger.info("Client started in simulation mode")
        else:
            self.logger.info("Client started with server %s".format(
                self.hname))

    def analyze_video_simulate(self, video_name, images_direc, batch_size,
                               high_results_path,
                               low_results_path, new_config=None):
        config_to_use = self.server_conf
        if new_config is not None:
            config_to_use = new_config

        results = Results()
        r1_results = Results()
        r2_results = Results()

        low_results_dict = read_results_dict(low_results_path, fmat="txt")
        self.logger.info("Reading low resolution results complete")
        high_results_dict = read_results_dict(high_results_path, fmat="txt")
        self.logger.info("Reading high resolution results complete")

        total_area_req_regions = 0
        number_of_frames = len(os.listdir(images_direc))
        for i in range(0, number_of_frames, batch_size):
            start_frame_id = i
            end_frame_id = min(number_of_frames, i + batch_size)
            self.logger.info("Processing batch from {} to {}".format(
                start_frame_id, end_frame_id))

            # Base (low resolution) phase
            r1, req_regions = self.server.simulate_low_query(start_frame_id,
                                                             end_frame_id,
                                                             images_direc,
                                                             low_results_dict,
                                                             config_to_use)
            self.logger.info("Got {} confirmed results with {} "
                             "regions to query in first phase of batch".format(
                                 r1.results_len(), req_regions.results_len()))
            r1_results.combine_results(r1)

            # Calculate area of regions required by the server
            area_req_regions = compute_area_of_regions(req_regions)
            total_area_req_regions += area_req_regions
            self.logger.info("{} regions have {} units total area".format(
                req_regions.results_len(), area_req_regions))

            # Second (high resolution) phase
            r2 = self.server.simulate_high_query(req_regions,
                                                 high_results_dict,
                                                 config_to_use)
            self.logger.info("Got {} results in second phase of batch".format(
                r2.results_len()))
            r2_results.combine_results(r2)

        # Combine results
        self.logger.info("Got {} unique results in base phase"
                         .format(r1_results.results_len()))
        results.combine_results(r1_results)
        self.logger.info("Got {} unique results in second phase"
                         .format(r2_results.results_len()))
        results.combine_results(r2_results)

        # Fill gaps in results
        results.fill_gaps(number_of_frames)

        # Write results
        write_results(results, video_name, fmat="txt")

        self.logger.info("Writing results for %s", video_name)
        self.logger.info("{} objects detected and {} total area "
                         "of regions sent in high resolution".format(
                             results.results_len(),
                             total_area_req_regions))

        return results, (0, total_area_req_regions)
