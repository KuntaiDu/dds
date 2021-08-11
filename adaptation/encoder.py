import os
import sys
from shutil import rmtree
from dds_utils import (read_results_dict, compute_regions_size,
                       Results, Region, calc_iou)

VID_NAME = "highway_0_00_00_00"
IMAGES_DIREC = "/home/ahsanp/Projects/video-analytics/new_dataset/highway_0_00_00_00/src"
N_FRAMES = len(sorted([e for e in os.listdir(IMAGES_DIREC) if "png" in e]))
RESULTS_DIREC = "results"
TMP_DIREC = "tmp"

PRUNE_SCORE = 0.5
SIZE_OBJ = 0.0025
OBJFILTER_IOU = 0.0
RPN_ENLARGE_RATIO = 0.0
SEG_SIZE = 15
FPS = 30


def get_regions_from_rpn(rpn_regions, detections):
    req_regions = Results()
    for region in rpn_regions.regions:
        # Continue if the size of region is too large
        if region.w * region.h > SIZE_OBJ:
            continue

        # If there are positive detections and they match a region
        # skip that region
        if len(detections) > 0:
            matches = 0
            for detection in detections.regions:
                if (calc_iou(detection, region) >
                        OBJFILTER_IOU and
                        detection.fid == region.fid and
                        region.label == 'object'):
                    matches += 1
            if matches > 0:
                continue

        # Enlarge and add to regions to be queried
        region.enlarge(RPN_ENLARGE_RATIO)
        req_regions.add_single_result(
            region, 1.0)
    return req_regions


def get_regions_to_query(start_fid, end_fid, lr, lq, hr, hq):
    results_path = os.path.join(RESULTS_DIREC, f"{VID_NAME}_mpeg_{lr}_{lq}")
    results_dict = read_results_dict(results_path)
    batch_results = Results()
    # Extract relevant results
    for fid in range(start_fid, end_fid):
        fid_results = results_dict[fid]
        for single_result in fid_results:
            single_result.origin = "low-res"
            batch_results.add_single_result(single_result, 1.0)

    detections = Results()
    rpn_regions = Results()
    # Divide RPN results into detections and RPN regions
    for single_result in batch_results.regions:
        if (single_result.conf > PRUNE_SCORE and
                single_result.label == "vehicle"):
            detections.add_single_result(
                single_result, 1.0)
        else:
            rpn_regions.add_single_result(
                single_result, 1.0)

    req_regions = get_regions_from_rpn(rpn_regions, detections)

    return req_regions


def get_size_with_config(start_fid, end_fid, lr, lq, hr, hq):
    # Encode for low
    base_req_regions = Results()
    for fid in range(start_fid, end_fid):
        base_req_regions.append(Region(fid, 0, 0, 1, 1, 1.0, 2, lr))
    low_size, _ = compute_regions_size(
        base_req_regions, f"{TMP_DIREC}-base-phase", IMAGES_DIREC, lr, lq,
        True, True)

    high_req_regions = get_regions_to_query(start_fid, end_fid, lr, lq, hr, hq)
    high_size, ps = compute_regions_size(
        high_req_regions, f"{TMP_DIREC}-high-phase", IMAGES_DIREC, hr, hq,
        True, True)

    rmtree(f"{TMP_DIREC}-base-phase-cropped")
    rmtree(f"{TMP_DIREC}-high-phase-cropped")

    return low_size, high_size


def read_config(lc, hc):
    lr, lq = None, None
    with open("lowConfiguration", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if int(line[0]) == lc:
                lr = float(line[1])
                lq = int(line[2])
    hr, hq = None, None
    with open("highConfiguration", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if int(line[0]) == hc:
                hr = float(line[1])
                hq = int(line[2])
    return lr, lq, hr, hq


def main(args):
    seg_num = int(args[1])
    lr, lq, hr, hq = read_config(int(args[2]), int(args[3]))

    start_fid = seg_num * SEG_SIZE
    end_fid = min((seg_num + 1) * SEG_SIZE, N_FRAMES)

    low_size, high_size = get_size_with_config(start_fid, end_fid,
                                               lr, lq, hr, hq)
    size = low_size + high_size
    low_bw = ((low_size * 8) / 1024) / ((end_fid - start_fid) / FPS)
    high_bw = ((high_size * 8) / 1024) / ((end_fid - start_fid) / FPS)
    # Calc bandwidth at FPS
    bw = ((size * 8) / 1024) / ((end_fid - start_fid) / FPS)
    with open("bw", "w") as f:
        f.write(f"{bw}")

    print(f"Processing {start_fid} - {end_fid} ({lr}, {lq}, {hr}, {hq}) for "
          f"{low_size}({low_bw}), {high_size}({high_bw}), {bw}",
          file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"USAGE: python {sys.argv[0]} "
              f"segment_number low_config high_config")
        exit()

    print(f"Running for {sys.argv}")
    main(sys.argv)
