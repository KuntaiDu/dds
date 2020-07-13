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

    args.low_resolution = args.resolutions[0]
    args.high_resolution = args.resolutions[1]
    args.low_qp = args.qp[0]
    args.high_qp = args.qp[1]

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
        mode = "simulation"
        logger.warning("Running DDS in SIMULATION mode")
        server = Server(config)

        logger.info("Starting client")
        client = Client(args.hname, config, server)
        # Run simulation
        logger.info(f"Analyzing video {args.video_name} with low resolution "
                    f"of {args.resolutions[0]} and high resolution of "
                    f"{args.resolutions[1]}")
        results, bw = client.analyze_video_simulate(
            args.video_name, args.low_images_path, args.high_images_path,
            args.high_results_path, args.low_results_path,
            args.enforce_iframes, args.mpeg_results_path,
            args.estimate_banwidth, args.debug_mode)
    elif not args.simulate and not args.hname and args.resolutions[-1] != -1:
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
                       f"{args.resolutions[0]} on {args.video_name}")
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
    if args.ground_truth:
        ground_truth_dict = read_results_dict(args.ground_truth)
        logger.info("Reading ground truth results complete")
        tp, fp, fn, _, _, _, f1 = evaluate(
            number_of_frames - 1, results.regions_dict, ground_truth_dict,
            args.low_threshold, 0.5, 0.4, 0.4)
        stats = (tp, fp, fn)
        logger.info(f"Got an f1 score of {f1} "
                    f"for this experiment {mode} with "
                    f"tp {stats[0]} fp {stats[1]} fn {stats[2]} "
                    f"with total bandwidth {sum(bw)}")
    else:
        logger.info("No groundtruth given skipping evalution")

    # Write evaluation results to file
    write_stats(args.outfile, f"{args.video_name}", config, f1,
                stats, bw, number_of_frames, mode)


if __name__ == "__main__":

    # load configuration dictonary from command line
    print(yaml.load(sys.argv[1], Loader=yaml.SafeLoader))
    args = munchify(yaml.load(sys.argv[1], Loader=yaml.SafeLoader))

    # If running simulation check if results files are present
    if (args.simulate and
            (args.low_results_path is None or args.high_results_path is None)):
        print("Low and high results files not given.\n"
              "Low and high results files "
              "are needed when running in simulation mode")
        exit()

    if (not args.simulate and not args.hname and len(args.resolutions) == 2):
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

    if not args.simulate and len(args.resolutions) == 2:
        if args.low_images_path:
            print("Discarding low images path")
            args.low_images_path = None
        args.intersection_threshold = 1.0

    if args.simulate and not args.low_images_path:
        print("Running simulation require low resolution images")
        exit()

    if len(args.resolutions) < 2:
        print("Only one resolution given, running MPEG emulation")
        args.intersection_threshold = 1.0
        args.resolutions.append(-1)
        if len(args.qp) == 2:
            args.qp[1] = -1
        else:
            args.qp.append(-1)
    else:
        if args.resolutions[1] < args.resolutions[0]:
            print("Given high resolution is less than low resolution, "
                  "swapping resolutions")
            args.resolutions[0], args.resolutions[1] = (args.resolutions[1],
                                                        args.resolutions[0])

    main(args)
