import os
from dds_utils import read_results_dict, Results
import matplotlib.pyplot as plt


emulation_direc = "../emulation-results"
results_dir = "experiment-data/results"
graphs_dir = "graphs"
vidnames = os.listdir(results_dir)


def count_dds(video):
    # read results
    results_dict = read_results_dict(os.path.join(results_dir, video),
                                     fmat="txt")
    sim_results = Results()
    for _, regions in results_dict.items():
        for region in regions:
            sim_results.add_single_result(region)

    # read ground truth results
    gt_dict = read_results_dict(os.path.join(emulation_direc,
                                             video, "GroundTruth", "Results"),
                                fmat="txt")
    gt_results = Results()
    for _, regions in gt_dict.items():
        for region in regions:
            if region.conf < 0.8:
                continue
            gt_results.add_single_result(region)

    # Count false negatives
    fn0, fn1, fn2 = 0, 0, 0
    for gt_region in gt_results.regions:
        matching_region = sim_results.is_dup(gt_region)

        if not matching_region:
            # No detection at all
            fn0 += 1
            continue

        if matching_region.conf > 0.8:
            # Not false negative
            continue

        if (matching_region.conf < 0.5 and
                "low-res" in matching_region.origin):
            # not high enough confidence to be checked
            fn1 += 1
        elif "high-res" in matching_region.origin:
            # server confirms mistake
            fn2 += 1

    # Count false positives
    fp0, fp1 = 0, 0
    for region in sim_results.regions:
        if region.conf < 0.8:
            # If not strong detection
            continue

        gt_results.is_dup(region)

        if gt_results.is_dup(region):
            # This is a true positive
            continue

        if "low-res" in region.origin:
            fp0 += 1
        elif "high-res" in region.origin:
            fp1 += 1

    return fn0, fn1, fn2, fp0, fp1


def count_low_high(video, resolution):
    # read results
    results_dict = read_results_dict(os.path.join(
        emulation_direc, video,
        "MPEG", resolution, "Results"),
                                     fmat="txt")
    results = Results()
    for _, regions in results_dict.items():
        for region in regions:
            results.add_single_result(region, 1.0)

    # read ground truth results
    gt_dict = read_results_dict(os.path.join(emulation_direc,
                                             video, "GroundTruth", "Results"),
                                fmat="txt")
    gt_results = Results()
    for _, regions in gt_dict.items():
        for region in regions:
            if region.conf < 0.8:
                continue
            gt_results.add_single_result(region, 1.0)

    # count false negatives
    fn0, fn1, fn2 = 0, 0, 0
    for region in gt_results.regions:
        matching_region = results.is_dup(region)

        if not matching_region:
            fn0 += 1
            continue

        if matching_region.conf > 0.8:
            continue

        if matching_region.conf < 0.5:
            fn1 += 1
        elif matching_region.conf > 0.5 and matching_region.conf < 0.8:
            fn2 += 1

    # count false positives
    fp = 0
    for region in results.regions:
        if region.conf < 0.8:
            # Skip weak detection
            continue

        if not gt_results.is_dup(region):
            fp += 1

    return fn0, fn1, fn2, fp


for video in vidnames:
    if "0113" not in video:
        continue

    fig, axs = plt.subplots(3, 3)

    fn0, fn1, fn2, fp0, fp1 = count_dds(video)
    # DDS
    axs[0, 0].pie([fn0, fn1, fn2], autopct="%1.1f%%",
                  radius=((fn0 + fn1 + fn2) / (fn0 + fn1 + fn2 + fp0 + fp1)))
    axs[0, 1].pie([fp0, fp1], autopct="%1.1f%%",
                  radius=((fp0 + fp1) / (fn0 + fn1 + fn2 + fp0 + fp1)))

    # Low resolution
    fn0, fn1, fn2, fp = count_low_high(video, "23_0.375")
    axs[1, 0].pie([fn0, fn1, fn2], autopct="%1.1f%%",
                  radius=((fn0 + fn1 + fn2) / (fn0 + fn1 + fn2 + fp)))
    axs[1, 1].pie([fp], autopct="%1.1f%%",
                  radius=((fp0 + fp1) / (fn0 + fn1 + fn2 + fp)))

    # High resolution
    fn0, fn1, fn2, fp = count_low_high(video, "23_0.75")
    axs[2, 0].pie([fn0, fn1, fn2], autopct="%1.1f%%",
                  radius=((fn0 + fn1 + fn2) / (fn0 + fn1 + fn2 + fp)))
    axs[2, 1].pie([fp], autopct="%1.1f%%",
                  radius=((fp0 + fp1) / (fn0 + fn1 + fn2 + fp)))

    # plt.show()
