import os
from dds_utils import read_results_dict, Results
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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

    # Count false positives and true positives
    tp0, tp1 = 0, 0
    fp0, fp1 = 0, 0
    for region in sim_results.regions:
        if region.conf < 0.8:
            # If not strong detection
            continue

        matching_region = gt_results.is_dup(region)

        if matching_region:
            # This is a true positive
            if "low-res" in region.origin:
                tp0 += 1
            elif "high-res" in region.origin:
                tp1 += 1
            gt_results.remove(matching_region)
            continue

        if "low-res" in region.origin:
            fp0 += 1
        elif "high-res" in region.origin:
            fp1 += 1

    return tp0, tp1, fn0, fn1, fn2, fp0, fp1


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

    # count false positives and true positives
    tp = 0
    fp = 0
    for region in results.regions:
        if region.conf < 0.8:
            # Skip weak detection
            continue

        matching_region = gt_results.is_dup(region)
        if not matching_region:
            fp += 1
        else:
            tp += 1
            gt_results.remove(matching_region)

    return tp, fn0, fn1, fn2, fp


def get_marks(pct, total):
    absolute_value = int(total * (pct / 100.0))
    return f"{pct:0.1f}%\n({absolute_value})"


pf = PdfPages("video-analysis.pdf")
for video in vidnames:
    if video.split("_")[-1] in ["0002", "0060", "0028", "0079"]:
        continue

    print(video)

    fig = plt.figure(figsize=(14, 8))
    axs = fig.subplots(3, 3)

    axs[0, 0].text(2.5, 2, video, weight="bold")
    axs[0, 0].text(-0.7, 1.5, "False Negatives", weight="bold")
    axs[0, 1].text(-0.7, 1.5, "False Positives", weight="bold")
    axs[0, 2].text(-0.8, 1.5, "True Positives", weight="bold")

    # Low resolution
    tp, fn0, fn1, fn2, fp = count_low_high(video, "23_0.375")
    if (fn0 + fn1 + fn2 + fp) == 0:
        continue
    rad = ((fn0 + fn1 + fn2) / (fn0 + fn1 + fn2 + fp + tp))
    pdist = 0.75 / rad
    wedges, texts, autotexts = axs[0, 0].pie([fn0, fn1, fn2],
                                             autopct=lambda pct: get_marks(pct, fn0 + fn1 + fn2),
                                             pctdistance=pdist, radius=rad)
    axs[0, 0].text(-2.5, 0, "Low", weight="bold")
    plt.setp(autotexts, size=8)
    rad = ((fp) / (fn0 + fn1 + fn2 + fp + tp))
    pdist = 0.75 / rad
    wedges, texts, autotexts = axs[0, 1].pie([fp], autopct=lambda pct: get_marks(pct, fp),
                                             pctdistance=pdist, radius=rad)
    plt.setp(autotexts, size=8)
    rad = ((tp) / (fn0 + fn1 + fn2 + fp + tp))
    pdist = 0.75 / rad
    wedges, texts, autotexts = axs[0, 2].pie([tp], autopct=lambda pct: get_marks(pct, tp),
                                             pctdistance=pdist, radius=rad)
    plt.setp(autotexts, size=8)

    # High resolution
    tp, fn0, fn1, fn2, fp = count_low_high(video, "23_0.75")
    if (fn0 + fn1 + fn2 + fp) == 0:
        continue
    rad = ((fn0 + fn1 + fn2) / (fn0 + fn1 + fn2 + fp + tp))
    pdist = 0.75 / rad
    wedges, texts, autotexts = axs[1, 0].pie([fn0, fn1, fn2],
                                             autopct=lambda pct: get_marks(pct, fn0 + fn1 + fn2),
                                             pctdistance=pdist,
                                             radius=rad)
    axs[1, 0].text(-2.5, 0, "High", weight="bold")
    plt.setp(autotexts, size=8)
    rad = ((fp) / (fn0 + fn1 + fn2 + fp + tp))
    pdist = 0.75 / rad
    wedges, texts, autotexts = axs[1, 1].pie([fp], autopct=lambda pct: get_marks(pct, fp),
                                             pctdistance=pdist,
                                             radius=rad)
    rad = ((tp) / (fn0 + fn1 + fn2 + fp + tp))
    pdist = 0.75 / rad
    wedges, texts, autotexts = axs[1, 2].pie([tp], autopct=lambda pct: get_marks(pct, tp),
                                             pctdistance=pdist,
                                             radius=rad)

    plt.setp(autotexts, size=8)

    tp0, tp1, fn0, fn1, fn2, fp0, fp1 = count_dds(video)
    if (fn0 + fn1 + fn2 + fp0 + fp1) == 0:
        continue
    # DDS
    rad = ((fn0 + fn1 + fn2) / (fn0 + fn1 + fn2 + fp0 + fp1 + tp0 + tp1))
    pdist = 0.75 / rad
    wedges, texts, autotexts = axs[2, 0].pie([fn0, fn1, fn2],
                                             autopct=lambda pct: get_marks(pct, fn0 + fn1 + fn2),
                                             pctdistance=pdist,
                                             radius=rad)
    axs[2, 0].legend(wedges, ["No detection", "c < 0.5", "Server confirms"],
                     loc="best",
                     bbox_to_anchor=(1, 0, 0.5, 1))
    axs[2, 0].text(-2.5, 0, "DDS", weight="bold")
    plt.setp(autotexts, size=8)
    rad = ((fp0 + fp1) / (fn0 + fn1 + fn2 + fp0 + fp1 + tp0 + tp1))
    pdist = 0.75 / rad
    wedges, texts, autotexts = axs[2, 1].pie([fp0, fp1], autopct=lambda pct: get_marks(pct, fp0 + fp1),
                                             pctdistance=pdist, radius=rad)
    plt.setp(autotexts, size=8)
    rad = ((tp0 + tp1) / (fn0 + fn1 + fn2 + fp0 + fp1 + tp0 + tp1))
    pdist = 0.75 / rad
    wedges, texts, autotexts = axs[2, 2].pie([tp0 + tp1], autopct=lambda pct: get_marks(pct, tp0 + tp1),
                                             pctdistance=pdist, radius=rad)
    plt.setp(autotexts, size=8)
    axs[2, 2].legend(wedges, ["True Positives"],
                     loc="best")

    pf.savefig()
    plt.close(fig)
pf.close()
