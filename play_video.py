import re
import logging
import argparse
from backend.server import Server
from frontend.client import Client
from dds_utils import (ServerConfig, read_results_dict,
                       evaluate, write_stats)


def main(args):
    logging.basicConfig(format="%(name)s -- %(levelname)s -- %(message)s",
                        level=args.verbosity.upper())

    logger = logging.getLogger("dds")
    logger.addHandler(logging.NullHandler())

    if args.hname is None:
        # Make backup ServerConfig
        config = ServerConfig(args.resolutions[0], args.resolutions[1],
                              args.high_threshold, args.low_threshold,
                              args.max_object_size, args.tracker_length,
                              args.boundary, args.intersection_threshold)

        # Make simulation objects
        logger.info(f"Starting server with high threshold of "
                    f"{args.high_threshold} low threshold of "
                    f"{args.low_threshold} tracker length of "
                    f"{args.tracker_length}")
        server = Server(config)
        logger.info("Starting client")
        client = Client(server, args.hname, config)

        # Run simulation
        logger.info(f"Analyzing video {args.video_name} with low resolution "
                    f"of {args.resolutions[0]} and high resolution of "
                    f"{args.resolutions[1]}")
        results, bw = client.analyze_video_simulate(args.video_name,
                                                    args.low_images_path,
                                                    args.bsize,
                                                    args.high_results_path,
                                                    args.low_results_path)
        low, high = bw

        # Evaluation and writing results
        # Read Groundtruth results
        ground_truth_dict = read_results_dict(args.ground_truth, fmat="txt")
        logger.info("Reading ground truth results complete")
        f1, stats = evaluate(results, ground_truth_dict, args.high_threshold)
        logger.info(f"Got an f1 score of {f1} "
                    f"for this experiment using simulation with "
                    f"tp {stats[0]} fp {stats[1]} fn {stats[2]} "
                    f"with total bandwidth {sum(bw)}")

        # Write evaluation results to file
        write_stats(args.outfile, args.video_name, args.bsize,
                    config, f1, stats, bw, fmat="txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze video using "
                                     "DDS protocol")
    # Video arguments
    parser.add_argument("--vid-name", dest="video_name",
                        required=True, type=str,
                        help="Name of video to analyze")
    parser.add_argument("--low-src", dest="low_images_path",
                        required=True, type=str,
                        help="Path of low resolution images of the video")
    parser.add_argument("--high-src", dest="high_images_path",
                        type=str, default=None,
                        help="Path to high resolution images of the video")
    parser.add_argument("--resolutions", dest="resolutions", type=float,
                        nargs=2, metavar=("LOW", "HIGH"), required=True,
                        help="The low resolution for DDS to use")
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
    parser.add_argument("--batch-size", dest="bsize",
                        type=int, default=15,
                        help="Segment size to use for DDS")
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
    parser.add_argument("--tracker-length", dest="tracker_length",
                        type=int, default=4,
                        help="Number of frame for tracking in ROI selection")
    parser.add_argument("--boundary",
                        type=float, default=0.2,
                        help="Size by which to enlarge boundary while "
                        "calculating regions to query")

    # Simulation settings
    parser.add_argument("--verbosity",
                        default="warning", type=str,
                        help="The verbosity of logging")
    parser.add_argument("--intersection-threshold",
                        dest="intersection_threshold",
                        default=0.5, type=float,
                        help="The intersection threshold to use"
                        " when combining results objects")

    args = parser.parse_args()

    # Check that results path given if running in simulation mode
    if (args.hname is None and
            (args.low_results_path is None or args.high_results_path is None)):
        print("Low and high results files not given.\n"
              "Low and high results files "
              "are needed when running in simulation mode")
        exit()

    if not re.match("DEBUG|INFO|WARNING|CRITICAL", args.verbosity.upper()):
        print("Incorrect argument for verbosity."
              "Verbosity can only be one of the following:\n"
              "\tdebug\n\tinfo\n\twarning\n\terror")
        exit()

    if args.hname and not args.high_images_loc:
        print("Need location of high resolution images when "
              "not running in simulation mode")
        exit()

    if args.estimate_banwidth and not args.high_images_loc:
        print("DDS needs location of high resolution images to "
              "calculate true bandwidth")
        exit()

    if args.resolutions[1] < args.resolutions[0]:
        logging.warn("Given high resolution is less than low resolution, "
                     "swapping resolutions")
        args.resolutions[0], args.resolutions[1] = (args.resolutions[1],
                                                    args.resolutions[0])

    main(args)
