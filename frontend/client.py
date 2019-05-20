import logging
import os
from dds_utils import (Results, read_results_dict, write_results,
                       ServerConfig, compute_area_of_regions)


class Client:
    """The client of the DDS protocol
       sends images in low resolution and waits for
       further instructions from the server. And finally receives results"""

    def __init__(self, server_handle, hname, config):
        self.hname = hname
        self.server = server_handle
        self.server_conf = config

        self.logger = logging.getLogger("client")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        if hname is None:
            self.logger.info("Client started in simulation mode")
        else:
            self.logger.info(f"Client started with server {self.hname}")

    def analyze_video_simulate(self, video_name, images_direc, batch_size,
                               high_results_path, low_results_path):
        results = Results()
        r1_results = Results()
        r2_results = Results()

        low_results_dict = read_results_dict(low_results_path, fmat="txt")
        self.logger.info("Reading low resolution results complete")
        high_results_dict = read_results_dict(high_results_path, fmat="txt")
        self.logger.info("Reading high resolution results complete")

        total_regions_count = 0
        total_area_req_regions = 0
        number_of_frames = len(os.listdir(images_direc))
        for i in range(0, number_of_frames, batch_size):
            start_frame_id = i
            end_frame_id = min(number_of_frames, i + batch_size)
            self.logger.info(f"Processing batch from {start_frame_id} "
                             f"to {end_frame_id}")

            # Base (low resolution) phase
            r1, req_regions = self.server.simulate_low_query(start_frame_id,
                                                             end_frame_id,
                                                             images_direc,
                                                             low_results_dict)
            self.logger.info(f"Got {r1.results_len()} confirmed "
                             f"results with {req_regions.results_len()} "
                             f"regions to query in first phase of batch")
            r1_results.combine_results(r1)

            # Add the number of regions requested by the server
            total_regions_count += req_regions.results_len()

            # Calculate area of regions required by the server
            area_req_regions = compute_area_of_regions(req_regions)
            total_area_req_regions += area_req_regions
            self.logger.info(f"{req_regions.results_len()} regions have "
                             f"{area_req_regions} units total area")

            # Second (high resolution) phase
            r2 = self.server.simulate_high_query(req_regions,
                                                 high_results_dict)
            self.logger.info(f"Got {r2.results_len()} results in "
                             f"second phase of batch")
            r2_results.combine_results(r2)

        # Combine results
        self.logger.info(f"Got {r1_results.results_len()} unique results "
                         f"in base phase")
        results.combine_results(r1_results)
        self.logger.info(f"Got {r2_results.results_len()} positive "
                         f"identifications out of {total_regions_count} "
                         f"requests in second phase")
        results.combine_results(r2_results)

        # Fill gaps in results
        results.fill_gaps(number_of_frames)

        # Write results
        write_results(results, video_name, fmat="txt")

        self.logger.info(f"Writing results for {video_name}")
        self.logger.info(f"{results.results_len()} objects detected "
                         f"and {total_area_req_regions} total area "
                         f"of regions sent in high resolution")

        return results, (0, total_area_req_regions)
