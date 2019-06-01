import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dds_utils import Results, read_results_dict, evaluate


emulation_direc = "../emulation-results"
sim_results_direc = "results"
high_threshold = 0.8
resolutions = ["0.375", "0.5", "0.625", "0.75"]

with open("stats", "r") as f:
    lines = f.readlines()
    lines = lines[1:]


pf = PdfPages("f1-difference.pdf")
experiments = os.listdir(sim_results_direc)
results_dict = {}
count = 0
for experiment in experiments:
    if experiment.split("_")[-1] in ["0060", "0079", "0027"]:
        continue

    vid = "_".join(experiment.split("_")[:-2])
    exp_res = "-".join(experiment.split("_")[-2:])

    fig = plt.Figure()
    ax = fig.add_subplot()

    gt_path = os.path.join(emulation_direc, vid, "GroundTruth", "Results")
    gt_dict = read_results_dict(gt_path, fmat="txt")
    gt_results = Results()
    for fid, regions in gt_dict.items():
        for r in regions:
            gt_results.add_single_result(r)

    sim_path = os.path.join(sim_results_direc, vid)
    sim_dict = read_results_dict(sim_path, fmat="txt")
    sim_results = Results()
    for fid, regions in sim_dict.items():
        for r in regions:
            sim_results.add_single_result(r)

    dds_f1, (tp, fp, fn) = evaluate(sim_results, gt_dict, high_threshold)
    bw = 0
    for line in lines:
        line = line.split(",")
        if experiment not in line[0]:
            continue
        bw = float(line[-1])

    print(dds_f1, bw * 8 / 1024 / 10)
    plt.scatter([bw * 8 / 1024 / 10], [dds_f1], marker="x", label=exp_res)
    if "DDS" not in results_dict:
        results_dict["DDS"] = np.array([0, 0, 0, 0, 0], dtype="float64")
    results_dict["DDS"] += np.array([dds_f1, tp, fp, fn, bw], dtype="float64")

    mpeg_results = []
    for res in resolutions:
        res_path = os.path.join(emulation_direc, vid, "MPEG",
                                f"23_{res}", "Results")
        res_dict = read_results_dict(res_path, fmat="txt")
        res_results = Results()
        for fid, regions in res_dict.items():
            for r in regions:
                res_results.add_single_result(r)
        f1, (tp, fp, fn) = evaluate(res_results, gt_dict, high_threshold)

        summary_path = os.path.join(emulation_direc, vid,
                                    "Result_log_MPEG.log")
        with open(summary_path, "r") as f:
            summary_lines = f.readlines()
        summary_lines = [l for l in summary_lines if l.rstrip().rstrip() != ""]
        bw = 0
        for idx, line in enumerate(summary_lines):
            if f"RES {res}" in line:
                bw = float(summary_lines[idx + 2])
                break
        size = bw * 1024.0 * 10.0

        mpeg_results.append((f1, size))
        if f"MPEG-{res}" not in results_dict:
            results_dict[f"MPEG-{res}"] = np.array([0, 0, 0, 0, 0],
                                                   dtype="float64")
        results_dict[f"MPEG-{res}"] += np.array([f1, tp, fp, fn, size],
                                                dtype="float64")

    f1s = [x[0] for x in mpeg_results]
    bws = [x[1] * 8 / 1024 / 10 for x in mpeg_results]

    plt.scatter(bws, f1s, label="MPEG")

    plt.title(vid)
    plt.legend(loc="lower right")
    plt.ylabel("F1")
    plt.ylim(0.5, 1)
    plt.xlabel("Bandwidth")

    pf.savefig()
    fig.clf()
    plt.close()
    count += 1

fig = plt.figure()
ax = fig.add_subplot(111)
# Draw aggregate DDS
total_f1, tp, fp, fn, size = results_dict["DDS"]
precision = tp / (fp + tp)
recall = tp / (fn + tp)
f1 = 2.0 * precision * recall / (precision + recall)
plt.scatter([size * 8 / 1024 / 10 / count], [f1], marker="x", label="DDS")

# Draw aggregate MPEG
mpeg_results = []
for label, (f1, tp, fp, fn, size) in results_dict.items():
    if "MPEG" not in label:
        continue

    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    f1 = 2.0 * precision * recall / (precision + recall)

    mpeg_results.append((f1, size))

f1s = [x[0] for x in mpeg_results]
bws = [x[1] * 8 / 1024 / 10 / count for x in mpeg_results]
plt.scatter(bws, f1s, label="MPEG")

plt.title("Aggregate (calculated using total TP, FP, FN)")
plt.ylabel("F1")
plt.ylim(0.5, 1)
plt.xlabel("Bandwidth (Kbps)")
plt.legend(loc="lower right")
pf.savefig()
fig.clf()
plt.close()

total_f1, tp, fp, fn, size = results_dict["DDS"]
plt.scatter([size * 8 / 1024 / 10 / count], [total_f1 / count], marker="x", label="DDS")

pf.close()
