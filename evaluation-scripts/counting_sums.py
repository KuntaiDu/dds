import matplotlib.pyplot as plt
import numpy as np
import os
from dds_utils import Results, Region, read_results_txt_dict

vidnames = [name for name in os.listdir("results")]

fn = [[], [], []]
fp = [[], []]

count = 0
for name in vidnames:
    fn0, fn1, fn2 = 0, 0, 0
    fp0, fp1, fp2 = 0, 0, 0
    print(os.path.join("results", name))

    results_dict = read_results_txt_dict(os.path.join("results", name))
    simulation_results = Results()
    for _, value in results_dict.items():
        for v in value:
            simulation_results.add_single_result(v)

    gt_dict = read_results_txt_dict(os.path.join("..", "emulation-results", name, "GroundTruth", "Results"))
    gt_results = Results()
    for _, value in gt_dict.items():
        for v in value:
            if v.conf < 0.8:
                continue
            gt_results.add_single_result(v)

    for r in gt_results.regions:
        dup_region = simulation_results.is_dup(r)
        if not dup_region:
            fn0 += 1
        elif dup_region.resolution == 0.375 and dup_region.conf < 0.5:
            fn1 += 1
        elif dup_region.resolution == 0.75 and dup_region.conf < 0.8:
            fn2 += 1

    fn[0].append(fn0)
    fn[1].append(fn1)
    fn[2].append(fn2)

    high_res_dict = read_results_txt_dict(os.path.join("..", "emulation-results", name, "MPEG", "23_0.75", "Results"))
    high_res_results = Results()
    for _, value in high_res_dict.items():
        for v in value:
            if v.conf < 0.8:
                continue
            high_res_results.add_single_result(v)

    total_fp = 0
    for r in simulation_results.regions:
        if r.conf < 0.8:
            continue
        dup_region = gt_results.is_dup(r)
        if dup_region:
            # continue if tp
            continue

        total_fp += 1
        if r.resolution == 0.375:
            if r.conf > 0.8:
                fp2 += 1

        if r.resolution == 0.75:
            if r.conf > 0.8:
                fp1 += 1

    fp0 = total_fp - (fp1 + fp2)

    fp[0].append(fp1)
    fp[1].append(fp2)

fig, axs = plt.subplots(1, 2)

print(fp)

fn[0] = sum(fn[0])
fn[1] = sum(fn[1])
fn[2] = sum(fn[2])

fp[0] = sum(fp[0])
fp[1] = sum(fp[1])

wedges, texts, autotexts = axs[0].pie(fn,
                                      autopct="%1.1f", shadow=True, textprops={'size': 'smaller'},
                                      radius=sum(fn) / (sum(fn) + sum(fp)) + 0.15)
plt.setp(autotexts, size="x-small")
axs[0].legend(wedges,
              ["No Detection", "Conf < 0.5", "Server confirms mistake"],
              loc="best")
axs[0].set_title("False Negatives")

wedges, texts, autotexts = axs[1].pie(fp,
                                      autopct="%1.1f",
                                      shadow=True,
                                      radius=sum(fp) / (sum(fn) + sum(fp)) - 0.1)
axs[1].legend(wedges,
              ["Server confirms mistake", "Conf > 0.8"],
              loc="best")
axs[1].set_title("False Positives")

plt.savefig("all-videos.svg")
