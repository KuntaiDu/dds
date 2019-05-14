import os
from dds_utils import Results, Region, read_results_txt_dict


vidnames = [name for name in os.listdir("results")]

fn = [[], [], []]
fp = [[], [], []]

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

    for r in simulation_results.regions:
        if r.conf < 0.8:
            continue
        dup_region = gt_results.is_dup(r)
        if dup_region:
            # continue if tp
            continue

        if r.resolution == 0.375:
            matching_high = high_res_results.is_dup(r)
            if not matching_high:
                fp2 += 1
            elif matching_high.conf < 0.8:
                fp0 += 1

        if r.resolution == 0.75:
            matching_high = high_res_results.is_dup(r)
            if not matching_high:
                fp2 += 1
            elif matching_high.conf > 0.8:
                fp1 += 1

    fp[0].append(fp0)
    fp[1].append(fp1)
    fp[2].append(fp2)

print(fp)
