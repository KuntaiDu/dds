import os
import re
import logging
from backend.server import Server
from frontend.client import Client
from dds_utils import (ServerConfig, read_results_dict,
                       evaluate, write_stats)
import sys

from munch import *
import yaml

def main(args):
    logging.basicConfig(
        format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s",
        level=args.verbosity.upper())

    logger = logging.getLogger("dds")
    logger.addHandler(logging.NullHandler())

    # Make simulation objects
    logger.info(f"Starting server with high threshold of "
                f"{args.high_threshold} low threshold of "
                f"{args.low_threshold} tracker length of "
                f"{args.tracker_length}")

    config = args

    # config = ServerConfig(
    #     args.resolutions[0], args.resolutions[1], args.qp[0], args.qp[1],
    #     args.batch_size, args.high_threshold, args.low_threshold,
    #     args.max_object_size, args.min_object_size, args.tracker_length,
    #     args.boundary, args.intersection_threshold, args.tracking_threshold,
    #     args.suppression_threshold, args.simulate, args.rpn_enlarge_ratio,
    #     args.prune_score, args.objfilter_iou, args.size_obj)

    server = None
    mode = None
    results, bw = None, None
    if args.simulate:
        raise NotImplementedError("We do not support simulation anymore")
    elif not args.simulate and not args.hname and args.high_resolution != -1:
        mode = "emulation"
        logger.warning(f"Running DDS in EMULATION mode on {args.video_name}")
        server = Server(config)

        logger.info("Starting client")
        client = Client(args.hname, config, server)
        # Run emulation
        results, bw = client.analyze_video_emulate(
            args.video_name, args.high_images_path,
            args.enforce_iframes, args.low_results_path, args.debug_mode)
    elif not args.simulate and not args.hname:
        mode = "mpeg"
        logger.warning(f"Running in MPEG mode with resolution "
                       f"{args.low_resolution} on {args.video_name}")
        server = Server(config)

        logger.info("Starting client")
        client = Client(args.hname, config, server)
        results, bw = client.analyze_video_mpeg(
            args.video_name, args.high_images_path, args.enforce_iframes)
    elif not args.simulate and args.hname:
        mode = "implementation"
        logger.warning(
            f"Running DDS using a server client implementation with "
            f"server running on {args.hname} using video {args.hname}")
        logger.info("Starting client")
        client = Client(args.hname, config, server)
        results, bw = client.analyze_video(
            args.video_name, args.high_images_path, config,
            args.enforce_iframes)

    # Evaluation and writing results
    # Read Groundtruth results
    low, high = bw
    f1 = 0
    stats = (0, 0, 0)
    number_of_frames = len(
        [x for x in os.listdir(args.high_images_path) if "png" in x])
    # if args.ground_truth:
    #     ground_truth_dict = read_results_dict(args.ground_truth)
    #     logger.info("Reading ground truth results complete")
    #     tp, fp, fn, _, _, _, f1 = evaluate(
    #         number_of_frames - 1, results.regions_dict, ground_truth_dict,
    #         args.low_threshold, 0.5, 0.4, 0.4)
    #     stats = (tp, fp, fn)
    #     logger.info(f"Got an f1 score of {f1} "
    #                 f"for this experiment {mode} with "
    #                 f"tp {stats[0]} fp {stats[1]} fn {stats[2]} "
    #                 f"with total bandwidth {sum(bw)}")
    # else:
    #     logger.info("No groundtruth given skipping evalution")

    # Write evaluation results to file
    write_stats(args.outfile, f"{args.video_name}", config, f1,
                stats, bw, number_of_frames, mode)


if __name__ == "__main__":

    # load configuration dictonary from command line
    # use munch to provide class-like accessment to python dictionary
    args = munchify(yaml.load(sys.argv[1], Loader=yaml.SafeLoader))

    if not args.simulate and not args.hname and args.high_resolution != -1:
        if not args.high_images_path:
            print("Running DDS in emulation mode requires raw/high "
                  "resolution images")
            exit()

    if not re.match("DEBUG|INFO|WARNING|CRITICAL", args.verbosity.upper()):
        print("Incorrect argument for verbosity."
              "Verbosity can only be one of the following:\n"
              "\tdebug\n\tinfo\n\twarning\n\terror")
        exit()

    if args.estimate_banwidth and not args.high_images_path:
        print("DDS needs location of high resolution images to "
              "calculate true bandwidth estimate")
        exit()

    if not args.simulate and args.high_resolution != -1:
        if args.low_images_path:
            print("Discarding low images path")
            args.low_images_path = None
        args.intersection_threshold = 1.0

    if args.method != "dds":
        assert args.high_resolution == -1, "Only dds support two quality levels"
            

    if args.high_resolution == -1:
        print("Only one resolution given, running MPEG emulation")
        assert args.high_qp == -1, "MPEG emulation only support one QP"
    else:
        assert args.low_resolution <= args.high_resolution, \
                f"The resolution of low quality({args.low_resolution})"\
                f"can't be larger than high quality({args.high_resolution})"
        assert not(args.low_resolution == args.high_resolution and 
                    args.low_qp < args.high_qp),\
                f"Under the same resolution, the QP of low quality({args.low_qp})"\
                f"can't be higher than the QP of high quality({args.high_qp})"

    main(args)
