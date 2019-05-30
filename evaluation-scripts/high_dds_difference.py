import os
import math
import cv2 as cv
from dds_utils import Results, read_results_dict, write_results

emulation_direc = "../emulation-results"
sim_results_direc = "results"
analysis_direc = "analysis"
high_threshold = 0.8


def evaluate(results, gt_dict, high_threshold, iou_threshold=0.5):
    gt_results = Results()
    for k, v in gt_dict.items():
        for single_result in v:
            if single_result.conf < high_threshold:
                continue
            gt_results.add_single_result(single_result, iou_threshold)

    # Save regions count because the regions that match
    # will be removed from the gt_regions to ensure better
    # search speed
    gt_regions_count = gt_results.results_len()

    fp_regions = Results()
    fp = 0.0
    tp = 0.0
    fn = 0.0
    for a in results.regions:
        # Make sure that the region has a high confidence
        if a.conf < high_threshold:
            continue

        # Find match in gt_results
        matching_region = None
        for b in gt_results.regions:
            if a.is_same(b, iou_threshold):
                tp += 1.0
                matching_region = b
                break

        if matching_region:
            # Remove region from ground truth if
            # it has already matched with a region in results
            gt_results.remove(matching_region)
        else:
            fp += 1.0
            fp_regions.add_single_result(a)

    fn = gt_regions_count - tp
    fn_regions = Results()
    for region in gt_results.regions:
        fn_regions.add_single_result(region)

    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    f1 = 2.0 * precision * recall / (precision + recall)

    # Check in low confidence regions if there is a match for
    # the false negatives
    low_conf_tp = Results()
    for a in results.regions:
        if a.conf >= high_threshold:
            continue

        matching_region = None
        for b in gt_results.regions:
            if a.is_same(b, iou_threshold):
                low_conf_tp.add_single_result(a)
                matching_region = b

        if matching_region:
            gt_results.remove(matching_region)

    if math.isnan(f1):
        f1 = 0.0

    return f1, (tp, fp, fn), fp_regions, fn_regions, low_conf_tp


def overlay_bboxes(results_list, src, dst):
    results_list_final = []
    for idx, result in enumerate(results_list):
        c = [0, 0, 0]
        c[idx] = 255
        results_list_final.append((result, c))
    for image_name in os.listdir(src):
        fid = int(image_name.split(".")[0])
        # iterate over all regions to find matching fid
        all_regions = []
        for result, color in results_list_final:
            for r in result.regions:
                if r.fid == fid:
                    all_regions.append((r, color))

        if not all_regions:
            continue

        image_path = os.path.join(src, image_name)
        image_np = cv.imread(image_path)
        width = image_np.shape[1]
        height = image_np.shape[0]

        for r, c in all_regions:
            x0 = int(r.x * width)
            y0 = int(r.y * height)
            x1 = int((r.w * width) + x0)
            y1 = int((r.h * height) + y0)

            cv.rectangle(image_np, (x0, y0), (x1, y1), c, 2)

        image_write_path = os.path.join(dst, image_name)
        cv.imwrite(image_write_path, image_np)


vids = os.listdir(sim_results_direc)
for vid in vids:
    sim_dict = read_results_dict(
        os.path.join(sim_results_direc, vid), fmat="txt")
    sim_results = Results()
    for fid, regions in sim_dict.items():
        for r in regions:
            sim_results.add_single_result(r)

    high_dict = read_results_dict(
        os.path.join(emulation_direc, vid,
                     "MPEG", "23_0.75", "Results"), fmat="txt")
    high_results = Results()
    for fid, regions in high_dict.items():
        for r in regions:
            high_results.add_single_result(r)

    f1, (tp, fp, fn), fp_regions, fn_regions, low_conf_tp = evaluate(
        sim_results, high_dict, high_threshold)

    vid_analysis_direc = os.path.join(analysis_direc, vid)
    os.makedirs(vid_analysis_direc, exist_ok=True)

    # Write all regions
    fp_regions_path = os.path.join(vid_analysis_direc, "fp")
    write_results(fp_regions, fp_regions_path, fmat="txt")
    fn_regions_path = os.path.join(vid_analysis_direc, "fn")
    write_results(fn_regions, fn_regions_path, fmat="txt")
    low_conf_tp_path = os.path.join(vid_analysis_direc, "low_conf_tp")
    write_results(low_conf_tp, low_conf_tp_path, fmat="txt")

    original_images_direc = os.path.join(emulation_direc, vid, "0.375")
    overlayed_images_direc = os.path.join(vid_analysis_direc, "images")
    os.makedirs(overlayed_images_direc, exist_ok=True)
    overlay_bboxes([fp_regions, fn_regions, low_conf_tp],
                   original_images_direc, overlayed_images_direc)
