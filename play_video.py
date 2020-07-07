import os
import re
import logging
import argparse
from backend.server import Server
from frontend.client import Client
from dds_utils import (ServerConfig, read_results_dict,
                       evaluate, write_stats)


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

    config = ServerConfig(
        args.resolutions[0], args.resolutions[1], args.qp[0], args.qp[1],
        args.bsize, args.high_threshold, args.low_threshold,
        args.max_object_size, args.min_object_size, args.tracker_length,
        args.boundary, args.intersection_threshold, args.tracking_threshold,
        args.suppression_threshold, args.simulate, args.rpn_enlarge_ratio,
        args.prune_score, args.objfilter_iou, args.size_obj)

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
    parser = argparse.ArgumentParser(description="Analyze video using "
                                     "DDS protocol")
    # Video arguments
    parser.add_argument("--vid-name", dest="video_name",
                        required=True, type=str,
                        help="Name of video to analyze")
    parser.add_argument("--low-src", dest="low_images_path",
                        type=str,
                        help="Path of low resolution images of the video")
    parser.add_argument("--high-src", dest="high_images_path",
                        type=str, default=None,
                        help="Path to high resolution images of the video")
    parser.add_argument("--resolutions", dest="resolutions", type=float,
                        nargs="+", required=True,
                        help="The resolutions to use. If only one given "
                        "runs MPEG emulation")
    parser.add_argument("--hname",
                        type=str, default=None,
                        help="Host name for server "
                        "(simulation mode if not given)")
    parser.add_argument("--low-results", dest="low_results_path",
                        type=str, default=None,
                        help="Path to file containing low resolution results")
    parser.add_argument("--high-results", dest="high_results_path",
                        type=str, default=None,
                        help="Path to file containing high resolution results")
    parser.add_argument("--ground-truth", dest="ground_truth",
                        type=str, default=None,
                        help="File containing the ground_truth")
    parser.add_argument("--mpeg-result-summary", dest="mpeg_results_path",
                        type=str, default=None,
                        help="The path to the results.log file for MPEG that "
                        "the simulator can use to calculate low bandwidth "
                        "without enoding images to low resolution")
    parser.add_argument("--batch-size", dest="bsize",
                        type=int, default=15,
                        help="Segment size to use for DDS")
    parser.add_argument("--qp", dest="qp", nargs="+",
                        default=None, type=int,
                        help="QP to be used for encoding video")
    parser.add_argument("--enforce-iframes", action="store_true",
                        dest="enforce_iframes",
                        help="Flag to whether or not enfore only 1 "
                        "iframe per batch")
    parser.add_argument("--output-file",
                        dest="outfile", type=str, default="stats",
                        help="The name of the file to which to append "
                        "statistics about the experiment after evaluation")
    parser.add_argument("--estimate-bandwidth",
                        dest="estimate_banwidth", action="store_true",
                        help="Flag to indicate whether DDS should encode "
                        "and estimate bandwidth or just give fractions for "
                        "high resolution regions")

    # Server config arguments
    parser.add_argument("--low-threshold", dest="low_threshold",
                        type=float, default=0.1,
                        help="High threshold for ROI selection")
    parser.add_argument("--high-threshold", dest="high_threshold",
                        type=float, default=0.8,
                        help="Low threshold for ROI selection")
    parser.add_argument("--max-size", dest="max_object_size",
                        type=float, default=0.3,
                        help="Maximum size of object as fraction frame")
    parser.add_argument("--min-size", dest="min_object_size",
                        type=float, default=None,
                        help="Minimum object size to cosider")
    parser.add_argument("--tracker-length", dest="tracker_length",
                        type=int, default=4,
                        help="Number of frame for tracking in ROI selection")
    parser.add_argument("--boundary",
                        type=float, default=0.2,
                        help="Size by which to enlarge boundary while "
                        "calculating regions to query")
    parser.add_argument("--suppression-threshold",
                        dest="suppression_threshold", type=float, default=0.5,
                        help="The iou threshold to use during "
                        "non maximum suppression")
    parser.add_argument("--rpn_enlarge_ratio",
                        dest="rpn_enlarge_ratio", type=float, default=0.,
                        help="rpn_enlarge_ratio")
    parser.add_argument("--prune-score",
                        dest="prune_score", type=float, default=1.1,
                        help="prune_score")
    parser.add_argument("--objfilter-iou",
                        dest="objfilter_iou", type=float, default=1.1,
                        help="objfilter_iou")
    parser.add_argument("--size-obj",
                        dest="size_obj", type=float, default=1.1,
                        help="size_obj")
    # Simulation settings
    parser.add_argument("--verbosity",
                        default="warning", type=str,
                        help="The verbosity of logging")
    parser.add_argument("--tracking-intersection-threshold",
                        dest="tracking_threshold", default=0.3,
                        type=float,
                        help="The threshold to use when determining whether "
                        "tracked region is already in accpected results")
    parser.add_argument("--intersection-threshold",
                        dest="intersection_threshold",
                        default=1.0, type=float,
                        help="The intersection threshold to use"
                        " when combining results objects")
    parser.add_argument("--simulate",
                        dest="simulate", action="store_true",
                        help="If provided use the given high and low results "
                        "files to simulate actual DDS output")
    parser.add_argument("--debug-mode",
                        dest="debug_mode", action="store_true",
                        help="If provided the simulator does not delete "
                        "intermediate high resolution videos")

    args = parser.parse_args()

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
