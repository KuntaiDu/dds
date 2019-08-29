import logging
import os
import shutil
from dds_utils import (Results, read_results_dict, cleanup, Region,
                       compress_and_get_size, compute_regions_size,
                       get_size_from_mpeg_results, extract_images_from_video)
from merger import *

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
        number_of_frames = len(
            [f for f in os.listdir(raw_images_path) if ".png" in f])

        final_results = Results()
        total_size = 0
        import timeit
        start = timeit.default_timer()
        for i in range(0, number_of_frames, self.config.batch_size):
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

            self.logger.info(f"Detection {len(results)} regions for "
                             f"batch {start_frame} to {end_frame} with a "
                             f"total size of {batch_video_size / 1024}KB")

            final_results.combine_results(
                results, self.config.intersection_threshold)
            # Remove encoded video manually
            # shutil.rmtree(f"{video_name}-base-phase-cropped")
            total_size += batch_video_size
        stop = timeit.default_timer()
        print('video time: ', stop-start)

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


    def analyze_video_emulate(self, video_name, high_images_path,
                              enforce_iframes, low_results_path=None,
                              debug_mode=False):
        final_results = Results()
        low_phase_results = Results()
        high_phase_results = Results()
        all_req_regions = Results()

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

            # with cache
            # '''
            # filtered_req_regions = Results()
            # get fid list
            # fid_req_regions = set(sorted([r.fid for r in req_regions.regions]))
            # create dict of req_regions
            # if self.config.batch_size == 1000:
            # print('req_regions_origin', len(req_regions))
            #
            # if results_catched_last_batch:
            #     # compare according last 15 frames
            #     for r in req_regions.regions:
            #         cnt = 0
            #         for single_cached_region in results_catched_last_batch.regions:
            #             if calc_iou(r, single_cached_region) > 0.5:
            #                 cnt += 1
            #         if cnt == 0:
            #             # print('remove')
            #             req_regions.remove(r)
            # print('req_regions_after_removing', len(req_regions))
            #
            # req_regions_dict = {}
            #
            # for rr in req_regions.regions:
            #     if rr.fid not in req_regions_dict:
            #         req_regions_dict[rr.fid] = []
            #         # import pdb; pdb.set_trace()
            #     req_regions_dict[rr.fid].append(rr)
            #
            # for fid in range(start_fid, end_fid):
            #     if fid > 0 and fid < end_fid - 1:
            #         # has overlap with prev and curr
            #         if fid-1 in req_regions_dict and fid+1 in req_regions_dict:
            #             # create pairwise list
            #             for r_fid_prev in req_regions_dict[fid-1]:
            #                 for r_fid_post in req_regions_dict[fid+1]:
            #                     if calc_iou(r_fid_prev, r_fid_post) > 0:
            #                         cnt = 0
            #                         if fid in req_regions_dict:
            #                             for r_fid_curr in req_regions_dict[fid]:
            #                                 if calc_iou(r_fid_prev, r_fid_curr) > 0.4 or calc_iou(r_fid_post, r_fid_curr) > 0.4:
            #                                     cnt += 1
            #                         else:
            #                             req_regions_dict[fid] = []
            #                         if cnt == 0:
            #                             # smooth r_prev and r_post
            #                             # import pdb; pdb.set_trace()
            #                             x_ave = min(r_fid_prev.x, r_fid_post.x)
            #                             y_ave = min(r_fid_prev.y, r_fid_post.y)
            #                             w_ave = max(r_fid_prev.x + r_fid_prev.w, r_fid_post.x + r_fid_post.w) - x_ave
            #                             h_ave = max(r_fid_prev.y + r_fid_prev.h, r_fid_post.y + r_fid_post.h) - y_ave
            #                             req_regions.append(
            #                                 Region(fid,
            #                                        x_ave,
            #                                        y_ave,
            #                                        w_ave,
            #                                        h_ave,
            #                                        1.0, 'object', self.config.low_resolution))
            #
            #                             req_regions_dict[fid].append(
            #                                 Region(fid,
            #                                        x_ave,
            #                                        y_ave,
            #                                        w_ave,
            #                                        h_ave,
            #                                        1.0, 'object', self.config.low_resolution)
            #                             )
            results_for_regions = Results()
            results_for_pruning = Results()
            accepted_results = Results() #already recieve
            # get highly confidence area:
            for single_result in req_regions.regions:
                # if self.config.pruning:
                if single_result.conf > self.config.prune_score and single_result.label == 'vehicle':
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

                    # else:

            # for rr in req_regions.regions:
            #     import pdb; pdb.set_trace()
            #     if set([rr.fid -1, rr.fid, rr.fid+1]).issubset(fid_req_regions):
            #         rr_prev, rr_post = req_regions_dict[rr.fid -1], req_regions_dict[rr.fid+1]
            #         for r_prev in rr_prev:
            #             if calc_iou(r_prev, rr) > 0.:
            #                 filtered_req_regions.add_single_result(rr)
            #                 r_perv_to_add = r_prev.copy()
            #                 r_perv_to_add.fid = rr.fid
            #                 filtered_req_regions.add_single_result(r_perv_to_add)
            #                 continue
            #         for r_post in rr_post:
            #             if calc_iou(r_post, rr) > 0.:
            #                 r_post_to_add = r_post.copy()
            #                 r_post_to_add.fid = rr.fid
            #                 filtered_req_regions.add_single_result(rr)
            #                 filtered_req_regions.add_single_result(r_post_to_add)
            #                 continue
            # import pdb; pdb.set_trace()

                # fid_for_merge, +3
            # req_regions = filtered_req_regions
            # '''
            # remove bbox not in RPN results
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



            # import pdb; pdb.set_trace()
            # Cleanup for the next batch
            cleanup(video_name, debug_mode, start_fid, end_fid)
            # shutil.rmtree(low_images_path)
        # import pdb; pdb.set_trace()

        self.logger.info(f"Got {len(low_phase_results)} unique results "
                         f"in base phase")
        self.logger.info(f"Got {len(high_phase_results)} positive "
                         f"identifications out of {total_regions_count} "
                         f"requests in second phase")

        # Fill gaps in results
        final_results.fill_gaps(number_of_frames)

        # Write results
        final_results.write(f"{video_name}")
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
