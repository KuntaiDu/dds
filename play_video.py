import argparse
from backend.server import Server
from frontend.client import Client


def main(args):
    if args.hname is None:
        # Make simulation objects
        server = Server(args.high_threshold, args.low_threshold,
                        args.max_object_size, args.tracker_length)
        client = Client(server, args.hname, args.high_threshold,
                        args.low_threshold, args.max_object_size,
                        args.tracker_length)

        # Run simulation
        client.analyze_video_simulate(args.video_name, args.images_loc,
                                      args.bsize, args.high_results_path,
                                      args.low_results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze video using "
                                     "DDS protocol")
    # Video arguments
    parser.add_argument("--vid-name", dest="video_name",
                        required=True, type=str,
                        help="Name of video to analyze")
    parser.add_argument("--src", dest="images_loc",
                        required=True, type=str,
                        help="Path to frames of the video")
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
    parser.add_argument("--batch-size", dest="bsize",
                        type=int, default=15,
                        help="Segment size to use for DDS")
    # Server config arguments
    parser.add_argument("-ht", dest="high_threshold",
                        type=float, default=0.1,
                        help="High threshold for ROI selection")
    parser.add_argument("-lt", dest="low_threshold",
                        type=float, default=0.8,
                        help="Low threshold for ROI selection")
    parser.add_argument("-ms", dest="max_object_size",
                        type=float, default=0.3,
                        help="Maximum size of object as fraction frame")
    parser.add_argument("-tl", dest="tracker_length",
                        type=int, default=4,
                        help="Number of frame for tracking in ROI selection")

    args = parser.parse_args()

    # Check that results path given if running in simulation mode
    if (args.hname is None and
            (args.low_results_path is None or args.high_results_path is None)):
        print("Low and high results files not given.\n"
              "Low and high results files "
              "are needed when running in simulation mode")
        exit()

    main(args)
