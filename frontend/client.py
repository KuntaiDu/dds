import os
from ..utils import Results, read_results_dict, write_results, ServerConfig


class Client:
    """The client of the DDS protocol
       sends images in low resolution and waits for
       further instructions from the server. And finally receives results"""

    def __init__(self, server_handle, hname=None, bsize=15, high_threshold=0.8,
                 low_threshold=0.3, max_object_size=0.3, tracker_length=4):
        self.hname = hname
        self.server_handle = server_handle
        self.server_conf = ServerConfig(high_threshold, low_threshold,
                                        max_object_size, tracker_length)

    def analyze_video_simulate(self, video_name, images_direc,
                               high_resutls_path,
                               low_results_path, new_config=None):
        config_to_use = self.server_conf
        if new_config is not None:
            config_to_use = new_config

        results = Results()
        r1_results = Results()
        r2_results = Results()

        low_results_dict = read_results_dict(low_results_path, fmat="txt")
        high_results_dict = read_results_dict(high_resutls_path, fmat="txt")

        number_of_frames = os.listdir(images_direc)
        for i in range(0, number_of_frames, self.batch_size):
            start_frame_id = i
            end_frame_id = min(number_of_frames, i + self.batch_size)
            r1, req_regions = self.server.simulate_low_query(start_frame_id,
                                                             end_frame_id,
                                                             images_direc,
                                                             low_results_dict,
                                                             config_to_use)
            r1_results.combine_results(r1)

            r2 = self.server.simulate_high_query(req_regions,
                                                 high_results_dict,
                                                 config_to_use)
            r2_results.combine_results(r2)

        # Combine results and sort on fid
        results.combine_results(r1)
        results.combine_results(r2)

        # Fill gaps in results
        results.fill_gaps(number_of_frames)

        # Write results
        write_results(results, video_name, fmat="txt")
