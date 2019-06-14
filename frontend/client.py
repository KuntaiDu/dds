import logging
import os
from dds_utils import (Results, read_results_dict,
                       compress_and_get_size, compute_regions_size,
                       get_size_from_mpeg_results)


class Client:
    """The client of the DDS protocol
       sends images in low resolution and waits for
       further instructions from the server. And finally receives results"""

    def __init__(self, server_handle, hname, config):
        self.hname = hname
        self.server = server_handle
        self.config = config

        self.logger = logging.getLogger("client")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        if hname is None:
            self.logger.info("Client started in simulation mode")
        else:
            self.logger.info(f"Client started with server {self.hname}")

    def analyze_video_mpeg(self, video_name, images_path,
                           raw_images_path=None):
        number_of_frames = len(
            [f for f in os.listdir(images_path) if ".mp4" not in f])
        results = self.server.perform_detection(images_path)

        try:
            for fname in os.listdir(images_path):
                if ".mp4" in fname:
                    total_size = os.path.getsize(fname)
                    break
        except FileNotFoundError:
            self.logger.warn("Could not find video file in the "
                             "images directory. Encoding video "
                             "using raw images")
            if raw_images_path is None:
                self.logger.critical("Path to raw images was not given"
                                     "Could not calculate size of file "
                                     "to be sent")
                exit()
            else:
                total_size = compress_and_get_size(images_path, 0,
                                                   number_of_frames - 1,
                                                   self.config.low_resolution)

        self.logger.info(f"Detection {len(results)} regions for "
                         f"{number_of_frames} with a total size of "
                         f"{total_size / 1024}KB")

        # Fill gaps in results
        results.fill_gaps(number_of_frames)

        # Write results
        results.write(video_name)

        return results, total_size

    def analyze_video_emulate(self, vid_name, low_images_path,
                              high_images_path, bsize, low_results_path,
                              mpeg_results_path, debug_mode):
        return None, None

    def analyze_video_simulate(self, video_name, low_images_path,
                               high_images_path, batch_size,
                               high_results_path, low_results_path,
                               mpeg_results_path=None,
                               estimate_banwidth=False, debug_mode=False):
        results = Results()
        r1_results = Results()
        r2_results = Results()

        low_results_dict = read_results_dict(low_results_path)
        self.logger.info("Reading low resolution results complete")
        high_results_dict = read_results_dict(high_results_path)
        self.logger.info("Reading high resolution results complete")

        total_regions_count = 0
        total_size = [0, 0]
        number_of_frames = len(
            [x for x in os.listdir(low_images_path) if ".mp4" not in x])
        for i in range(0, number_of_frames, batch_size):
            start_frame_id = i
            end_frame_id = min(number_of_frames, i + batch_size)
            self.logger.info(f"Processing batch from {start_frame_id} "
                             f"to {end_frame_id}")

            # Base (low resolution) phase
            r1, req_regions = self.server.simulate_low_query(start_frame_id,
                                                             end_frame_id,
                                                             low_images_path,
                                                             low_results_dict)
            self.logger.info(f"Got {len(r1)} confirmed "
                             f"results with {len(req_regions)} "
                             f"regions to query in first phase of batch")
            r1_results.combine_results(r1, self.config.intersection_threshold)

            # Add the number of regions requested by the server
            total_regions_count += len(req_regions)

            encoded_batch_video_size = 0
            if not mpeg_results_path and estimate_banwidth:
                encoded_batch_video_size = compress_and_get_size(
                    high_images_path, start_frame_id, end_frame_id,
                    self.config.low_resolution, debug_mode)
                total_size[0] += encoded_batch_video_size

            regions_size = compute_regions_size(
                req_regions, video_name, high_images_path,
                self.config.high_resolution, estimate_banwidth)
            total_size[1] += regions_size
            self.logger.info(f"{encoded_batch_video_size}KB sent in base "
                             f"phase {len(req_regions)} regions have "
                             f"{regions_size} units total size")

            # Second (high resolution) phase
            r2 = self.server.simulate_high_query(req_regions,
                                                 high_results_dict)
            self.logger.info(f"Got {len(r2)} results in "
                             f"second phase of batch")
            r2_results.combine_results(r2, self.config.intersection_threshold)

        # Combine results
        self.logger.info(f"Got {len(r1_results)} unique results "
                         f"in base phase")
        results.combine_results(r1_results, self.config.intersection_threshold)
        self.logger.info(f"Got {len(r2_results)} positive "
                         f"identifications out of {total_regions_count} "
                         f"requests in second phase")
        results.combine_results(r2_results, self.config.intersection_threshold)

        # Fill gaps in results
        results.fill_gaps(number_of_frames)

        # Write results
        results.write(video_name)

        # Get results from summary file if given
        if mpeg_results_path and estimate_banwidth:
            total_size[0] = get_size_from_mpeg_results(
                mpeg_results_path, low_images_path, self.config.low_resolution)

        self.logger.info(f"Writing results for {video_name}")
        self.logger.info(f"{len(results)} objects detected "
                         f"and {total_size[1]} total size "
                         f"of regions sent in high resolution")

        return results, total_size
