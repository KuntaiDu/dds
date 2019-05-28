import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm


resolutions = ["0.375", "0.75"]

cm_colors = iter(cm.rainbow(np.linspace(0, 1, 35)))
colors = []
for i in range(0, 35):
    colors.append(next(cm_colors))

with open("stats", "r") as f:
    lines = f.readlines()
    lines = lines[1:]

results_dict = {}

for l in lines:
    line = l.split(',')
    experiment_string = line[0].split("/")[-1]
    vid_name = "_".join(experiment_string.split("_")[:-2])
    label = "-".join(experiment_string.split("_")[-2:])
    bw = float(line[-1])
    f1 = float(line[-4])

    if vid_name not in results_dict:
        results_dict[vid_name] = []

    results_dict[vid_name].append((f1, bw, label))

pp = PdfPages('test.pdf')
for vid_name, exps in results_dict.items():
    if vid_name.split("_")[-1] in ["0002"]:
        continue

    mpeg_results_file = os.path.join("..", "emulation-results",
                                     vid_name, "Result_log_MPEG.log")
    with open(mpeg_results_file, "r") as f:
        lines = f.readlines()

    mpeg_results = []
    for idx in range(0, len(lines), 5):
        curr_res = lines[idx].split(" ")[1].rstrip().lstrip()
        if curr_res in resolutions:
            bw = float(lines[idx + 2]) * 8
            f1 = float(lines[idx + 3].split(" ")[-1])
            mpeg_results.append((f1, bw))

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)

    plt.scatter([x[1] for x in mpeg_results], [x[0] for x in mpeg_results],
                marker="x")

    for idx, (f1, bw, label) in enumerate(exps):
        plt.scatter([bw * 8 / 1024 / 10], [f1], label=label, color=colors[idx])

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0,
                     chartBox.width*0.7, chartBox.height])

    plt.legend(loc="upper center", bbox_to_anchor=(1.3, 0.8), ncol=4)

    plt.title(vid_name)
    plt.xlabel("Bandwidth (Kbps)")
    plt.ylabel("F1")

    pp.savefig()

    plt.close(fig)

pp.close()
