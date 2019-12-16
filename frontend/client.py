import logging
import os
import shutil
from dds_utils import (Results, read_results_dict, cleanup, Region,
                       compress_and_get_size, compute_regions_size,
                       get_size_from_mpeg_results, extract_images_from_video)
from merger import *

import yaml
import pickle
with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())
relevant_classes = dds_env['relevant_classes']

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

        self.logger.info(f"Client initialized")

    def analyze_video_mpeg(self, video_name, raw_images_path, enforce_iframes):
        if dds_env['application'] == 'detection':
            return self.analyze_video_mpeg_detection(video_name, raw_images_path, enforce_iframes)
        elif dds_env['application'] == 'classification':
            return self.analyze_video_mpeg_classification(video_name, raw_images_path, enforce_iframes)

    def analyze_video_mpeg_detection(self, video_name, raw_images_path, enforce_iframes):
        number_of_frames = len(
            [f for f in os.listdir(raw_images_path) if ".png" in f])

        final_results = Results()
        total_size = 0
        segment_size_file = open(f'{video_name}_segment_size', 'w')
        for i in range(0, number_of_frames, self.config.batch_size):
            import timeit
            start = timeit.default_timer()
            start_frame = i
            end_frame = min(number_of_frames, i + self.config.batch_size)
            print(start_frame, end_frame)

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
            results, multi_scan_final_results, offsets = self.server.perform_detection(
                f"{video_name}-base-phase-cropped", self.config.low_resolution,
                batch_fnames)
            stop = timeit.default_timer()
            batch_time =  stop-start

            self.logger.info(f"Detection {len(results)} regions for "
                             f"batch {start_frame} to {end_frame} with a "
                             f"total size of {batch_video_size / 1024}KB")

            final_results.combine_results(
                results, self.config.intersection_threshold)
            # Remove encoded video manually
            shutil.rmtree(f"{video_name}-base-phase-cropped")
            total_size += batch_video_size
            segment_size_file.write(f"{batch_video_size}, {batch_time}\n")
        segment_size_file.close()

        # Fill gaps in results
        final_results.fill_gaps(number_of_frames)

        # Write results
        final_results.write(video_name)
        # do simple merge here
        # import pdb; pdb.set_trace()
        # print(len(final_results))
        rdict = read_results_dict(video_name)
        final_results = merge_boxes_in_results(rdict, 0.3, 0.3)
        final_results.fill_gaps(number_of_frames)
        # print(len(final_results))
        final_results.write(video_name)

        return final_results, [total_size, 0]

    def analyze_video_mpeg_classification(self, video_name, raw_images_path, enforce_iframes):
        number_of_frames = len(
            [f for f in os.listdir(raw_images_path) if ".png" in f])

        total_size = 0
        segment_size_file = open(f'{video_name}_segment_size', 'w')

        results = {}

        for i in range(0, number_of_frames, self.config.batch_size):
            import timeit
            start = timeit.default_timer()
            start_frame = i
            end_frame = min(number_of_frames, i + self.config.batch_size)
            print(start_frame, end_frame)

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
            self.server.perform_classification(
                f"{video_name}-base-phase-cropped", self.config.low_resolution,
                batch_fnames, results = results)
            stop = timeit.default_timer()
            batch_time = stop-start

            self.logger.info(f"Perform classification for "
                             f"batch {start_frame} to {end_frame} with a "
                             f"total size of {batch_video_size / 1024}KB")

            # Remove encoded video manually
            shutil.rmtree(f"{video_name}-base-phase-cropped")
            total_size += batch_video_size
            segment_size_file.write(f"{batch_video_size}, {batch_time}\n")

        with open(video_name, 'wb') as f:
            pickle.dump(results, f)

        return results, [total_size, 0]

    def analyze_video_emulate(
        self,
        video_name,
        high_images_path,
        enforce_iframes,
        low_results_path = None,
        debug_mode = None):

        if dds_env['application'] == 'detection':
            return self.analyze_video_emulate_detection(
                video_name,
                high_images_path,
                enforce_iframes,
                low_results_path,
                debug_mode)
        elif dds_env['application'] == 'classification':
            return self.analyze_video_emulate_classification(
                video_name,
                high_images_path,
                enforce_iframes,
                low_results_path,
                debug_mode
            )


    def analyze_video_emulate_detection(self, video_name, high_images_path,
                              enforce_iframes, low_results_path=None,
                              debug_mode=False):
        final_results = Results()
        low_phase_results = Results()
        high_phase_results = Results()
        all_req_regions = Results()
        final_r1_results = Results()
        final_r2_results = Results()

        ##################################
        segment_size_file = open(f'{video_name}_segment_size', 'w')

        number_of_frames = len(
            [x for x in os.listdir(high_images_path) if "png" in x])

        low_results_dict = None
        if low_results_path:
            low_results_dict = read_results_dict(low_results_path)

        total_size = [0, 0]
        total_pixel_size = 0
        total_regions_count = 0
        results_catched_last_batch = None
        for i in range(0, number_of_frames, self.config.batch_size):
            accepted_r2_results = Results()
            accepted_r1_results = Results()
            # print(i)
            start_fid = i
            end_fid = min(number_of_frames, i + self.config.batch_size)
            self.logger.info(f"Processing batch from {start_fid} to {end_fid}")

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
                enforce_iframes, True, 1)
            # import pdb; pdb.set_trace()
            # self.logger.info(f"{encoded_batch_video_size / 1024}KB sent "
            #                  f"in base phase using {self.config.low_qp}")
            total_size[0] += encoded_batch_video_size

            # Low resolution phase
            low_images_path = f"{video_name}-base-phase-cropped"
            r1, req_regions = None, None
            if low_results_dict:
                # If results dict is present then just simulate the first phase
                extract_images_from_video(low_images_path, base_req_regions)
                req_regions = (
                    self.server.simulate_low_query(start_fid, end_fid,
                                                   low_images_path,
                                                   low_results_dict, False, self.config.rpn_enlarge_ratio))
                # for r in req_regions.regions:
                #     all_req_regions.append(r)
            else:
                # # If results dict is not present then actually
                # # emulate first phase
                # r1, req_regions = (
                #     self.server.emulate_low_query(start_fid, end_fid,
                #                                   low_images_path,
                #                                   req_regions))
                import pdb; pdb.set_trace()
                # for r in req_regions.regions:
                #     all_req_regions.append(r)
            # self.logger.info(f"Got {len(r1)} confirmed regions with  "
            #                  f"{len(req_regions)} regions to query in "
            #                  f"the first phase")
            low_phase_results.combine_results(
                all_req_regions, self.config.intersection_threshold)
            # final_results.combine_results(
            #     all_req_regions, self.config.intersection_threshold)
            # import pdb; pdb.set_trace()
            total_regions_count += len(req_regions)

            results_for_regions = Results()
            results_for_pruning = Results()
            accepted_results = Results() #already recieve
            # get highly confidence area:
            for single_result in req_regions.regions:
                # if self.config.pruning:
                if single_result.conf > self.config.prune_score and single_result.label in relevant_classes:
                    # these for sure
                    accepted_results.add_single_result(
                        single_result, self.config.intersection_threshold)

                    continue
                results_for_pruning.add_single_result(single_result, self.config.intersection_threshold)

            print('results_for_pruning', len(results_for_pruning))
            print('accepted_results', len(accepted_results))

            for single_result in results_for_pruning.regions:
                # find bbox to do pruning:
                if len(accepted_results) > 0:
                    cnt = 0
                    for single_accept_result in accepted_results.regions:
                        if calc_iou(single_accept_result, single_result) > self.config.objfilter_iou \
                            and single_accept_result.fid == single_result.fid \
                            and single_result.label == 'object':
                            cnt += 1
                    if cnt > 0:
                        continue
                if single_result.w * single_result.h > self.config.size_obj:
                    continue
                x_min = max(single_result.x - single_result.w * self.config.rpn_enlarge_ratio, 0.)
                y_min = max(single_result.y - single_result.h * self.config.rpn_enlarge_ratio, 0.)
                x_max = min(single_result.x + single_result.w * (1 + self.config.rpn_enlarge_ratio), 1.)
                y_max = min(single_result.y + single_result.h * (1 + self.config.rpn_enlarge_ratio), 1.)
                single_result.x = x_min
                single_result.y = y_min
                single_result.w = x_max - x_min
                single_result.h = y_max - y_min
                results_for_regions.add_single_result(
                    single_result, self.config.intersection_threshold)
                        # filter req_region
            req_regions = results_for_regions
            print('req_regions', len(req_regions))
            for r in req_regions.regions:
                all_req_regions.append(r)

            if len(req_regions) > 0:
                # Crop, compress and get size
                regions_size, pixel_size = compute_regions_size(
                    req_regions, video_name, high_images_path,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True, 1)
                self.logger.info(f"Sent {len(req_regions)} regions which have "
                                 f"{regions_size / 1024}KB in second phase "
                                 f"using {self.config.high_qp}")
                total_size[1] += regions_size
                total_pixel_size += pixel_size
                # import pdb; pdb.set_trace()
                # High resolution phase
                # every three filter
                r2 = self.server.emulate_high_query(
                    video_name, low_images_path, req_regions)
                results_catched_last_batch = r2
                self.logger.info(f"Get {len(r2)} results in second phase "
                                 f"of batch")
                high_phase_results.combine_results(
                    r2, self.config.intersection_threshold)
                # if self.config.pruning:
                if len(accepted_results) > 0:
                    final_results.combine_results(
                        accepted_results, self.config.intersection_threshold)
                final_results.combine_results(
                    r2, self.config.intersection_threshold)
                for single_result in r2.regions:
                    # find bbox to do pruning:
                    if single_result.conf < self.config.prune_score:
                        continue
                    if len(req_regions) > 0:
                        cnt = 0
                        for single_req_result in req_regions.regions:
                            # this is for processing delay calc
                            # only acc
                            if calc_iou(single_req_result, single_result) > 0.8 \
                                and single_req_result.fid == single_result.fid:
                                cnt += 1
                        if cnt > 0:
                            accepted_r2_results.add_single_result(single_result, self.config.intersection_threshold)
            segment_size_file.write(f"{encoded_batch_video_size},{regions_size},{len(accepted_results)},{len(accepted_r2_results)}\n")
            final_r1_results.combine_results(accepted_results,1.0)
            final_r2_results.combine_results(accepted_r2_results,1.0)
            # import pdb; pdb.set_trace()
            # Cleanup for the next batch
            cleanup(video_name, debug_mode, start_fid, end_fid)
            shutil.rmtree(low_images_path)
        # import pdb; pdb.set_trace()

        segment_size_file.close()
        self.logger.info(f"Got {len(low_phase_results)} unique results "
                         f"in base phase")
        self.logger.info(f"Got {len(high_phase_results)} positive "
                         f"identifications out of {total_regions_count} "
                         f"requests in second phase")

        # Fill gaps in results
        final_results.fill_gaps(number_of_frames)
        final_r1_results.fill_gaps(number_of_frames)
        final_r2_results.fill_gaps(number_of_frames)

        # Write results
        final_results.write(f"{video_name}")
        final_r1_results.write(f"{video_name}_r1_res")
        final_r2_results.write(f"{video_name}_r2_res")
        all_req_regions.write(
            f"{video_name}_req_regions_"
            f"{self.config.low_threshold}_{self.config.high_threshold}_{self.config.rpn_enlarge_ratio}")

        self.logger.info(f"Writing results for {video_name}")
        self.logger.info(f"{len(final_results)} objects detected "
                         f"and {total_size[1]} total size "
                         f"of regions sent in high resolution")

        # # do simple merge here
        # # import pdb; pdb.set_trace()
        # print(len(final_results))
        rdict = read_results_dict(f"{video_name}")
        # MONKEY CODE
        final_results = merge_boxes_in_results(rdict, 0.3, 0.3)
        # MONKEY CODE

        final_results.fill_gaps(number_of_frames)
        # print(len(final_results))
        final_results.write(f"{video_name}")
        return final_results, total_size


    def analyze_video_emulate_classification(self, video_name, high_images_path,
                              enforce_iframes, low_results_path=None,
                              debug_mode=False):

        # segment_size_file = open(f'{video_name}_segment_size', 'w')

        number_of_frames = len(
            [x for x in os.listdir(high_images_path) if "png" in x])
        low_results_dict = read_results_dict(low_results_path)
        total_size = [0, 0]


        # count size for low config
        for i in range(0, number_of_frames, self.config.batch_size):

            start_fid = i
            end_fid = min(number_of_frames, i + self.config.batch_size)
            self.logger.info(f'Processing batch from {start_fid} to {end_fid}')

            # config paths
            low_images_path = f"{video_name}-base-phase-cropped"

            # encode low quality
            base_req_regions = Results()
            for fid in range(start_fid, end_fid):
                base_req_regions.append(
                    Region(fid - start_fid, 0, 0, 1, 1, 1.0, "pass",
                           self.config.low_resolution))
            encoded_batch_video_size, _ = compute_regions_size(
                base_req_regions, f"{video_name}-base-phase", high_images_path,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True, 1)
            total_size[0] += encoded_batch_video_size

            # encode mixed quality
            high_req_regions = Results()
            for fid in range(start_fid, end_fid):
                for region in low_results_dict[fid]:
                    high_req_regions.append(Region(
                            fid - start_fid,
                            region.x,
                            region.y,
                            region.w,
                            region.h,
                            1.0,
                            "pass",
                            self.config.high_resolution))
            regions_size, _ = compute_regions_size(
                high_req_regions,
                video_name,
                high_images_path,
                self.config.high_resolution,
                self.config.high_qp,
                enforce_iframes,
                True,
                1
            )
            total_size[1] += regions_size

            # calculate performance
            classification_results = self.server.emulate_high_query(
                video_name,
                low_images_path,
                high_req_regions
            )

            cleanup(video_name, debug_mode, start_fid, end_fid)
            shutil.rmtree(low_images_path)

        # save the result
        with open(f"{video_name}", 'wb') as f:
            pickle.dump(classification_results, f)

        return classification_results, total_size

