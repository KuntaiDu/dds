import os


data_dir = "/data/yuanx/new_dataset"
gt_qps = {
    "manhattan_021_00_00_00": 34,
    "flyover_4_00_00_00": 32,
    "flyover_2_00_00_00": 30,
    "highway_1_00_00_00": 31,
    "highway_0_00_00_00": 24,
    "highway_1_45_00_00": 31,
    "ladash_002_00_00_00": 34,
    "ladash_000_00_00_00": 33,
    "street_0_00_00_07": 30,
    "flyover_1_00_00_00": 31,
    "interchange_0_00_00_1": 30,
    "interchange_1_00_00_00": 29,
    "highway_1_24_00_00": 31,
    "interchange_0_00_00_00": 30,
    "highway_1_47_00_00": 31,
    "twoway_0_00_00_00": 27,
}
mpeg_resolutions = [0.375, 0.5, 0.625, 0.7, 0.75, 0.8, 0.9]
mpeg_qps = {
    "manhattan_021_00_00_00": [35, 36, 37, 38, 39, 40],
    "flyover_4_00_00_00": [34, 35, 36, 37, 38, 39, 40],
    "flyover_2_00_00_00": [33, 35, 36, 37, 38, 39, 40],
    "highway_1_00_00_00": [33, 35, 36, 37, 38, 39, 40],
    "highway_0_00_00_00": [26, 28, 30, 33, 35, 36, 37, 38, 39, 40],
    "highway_1_45_00_00": [36, 40],
    "ladash_002_00_00_00": [35, 36, 37, 38, 39, 40],
    "ladash_000_00_00_00": [35, 36, 37, 38, 39, 40],
    "street_0_00_00_07": [33, 35, 36, 37, 38, 39, 40],
    "flyover_1_00_00_00": [33, 35, 36, 37, 38, 39, 40],
    "interchange_0_00_00_1": [33, 35, 36, 37, 38, 39, 40],
    "interchange_1_00_00_00": [32, 35, 36, 37, 38, 39, 40],
    "highway_1_24_00_00": [33, 35, 36, 37, 38, 39, 40],
    "interchange_0_00_00_00": [33, 35, 36, 37, 38, 39, 40],
    "highway_1_47_00_00": [33, 35, 36, 37, 38, 39, 40],
    "twoway_0_00_00_00": [29, 33, 35, 37, 38, 39, 40]
}
dds_config = {
    "manhattan_021_00_00_00": [(0.2, 39, 0.25, 35), (0.2, 36, 0.625, 38)],
    "flyover_4_00_00_00": [(0.25, 40, 0.5, 40), (0.25, 39, 0.5, 40)],
    "flyover_2_00_00_00": [(0.25, 40, 0.375, 40), (0.375, 40, 0.625, 38)],
    "highway_1_00_00_00": [(0.25, 40, 0.5, 40)],
    "highway_0_00_00_00": [(0.375, 40, 0.75, 36)],
    "out10fps720p": [(0.5, 40, 0.75, 36), (0.375, 40, 0.75, 36), (0.375, 40, 0.9, 32), (0.5, 40, 0.9, 32), (0.5, 40, 0.8, 32), (0.5, 40, 0.9, 36)],
    "ladash_002_00_00_00": [(0.25, 40, 0.5, 40), (0.25, 39, 0.5, 40)],
    "ladash_000_00_00_00": [(0.25, 39, 0.625, 40), (0.25, 38, 0.625, 40)],
    "street_0_00_00_07": [(0.375, 40, 0.625, 40), (0.25, 39, 0.625, 40)],
    "flyover_1_00_00_00": [(0.2, 37, 0.625, 40), (0.25, 38, 0.625, 40), (0.25, 37, 0.625, 40)],
    "interchange_0_00_00_1": [(0.375, 40, 0.5, 36), (0.25, 39, 0.5, 39)],
    "interchange_1_00_00_00": [(0.375, 40, 0.5, 38)],
    "highway_1_24_00_00": [(0.25, 39, 0.5, 40)],
    "interchange_0_00_00_00": [(0.375, 40, 0.5, 38)],
    "highway_1_47_00_00": [(0.2, 40, 0.375, 40), (0.25, 40, 0.5, 40)],
    "twoway_0_00_00_00": []
}
vid_names = sorted(gt_qps.keys())

for video in vid_names:
    # Generate gt
    if video != "drive_sing_cut10fps":
        continue
    res = 1.0
    qp = gt_qps[video]
    vname = f"{video}_gt"
    if not os.path.exists(os.path.join("results", vname)):
        os.system(f"bash ./experiment-scripts/run_single.sh gt {video} {vname} {qp}")
    else:
        print(f"Skipping {vname}")

    # Run MPEG
    for res in mpeg_resolutions:
        for qp in mpeg_qps[video]:
            print(str(res) + '/' + str(qp))
            vname = f"{video}_mpeg_{res}_{qp}"
            if not os.path.exists(os.path.join("results", vname)):
                os.system(f"bash ./experiment-scripts/run_single.sh mpeg {video} {vname} {qp} {res}")
            else:
                print(f"Skipping {vname}")

    # Run DDS
    for low_res, low_qp, high_res, high_qp in dds_config[video]:
        vname = f"{video}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}"
        if not os.path.exists(os.path.join("results", vname)):
            os.system(f"bash ./experiment-scripts/run_single.sh dds {video} {vname} {low_qp} {high_qp} {low_res} {high_res}")
        else:
            print(f"Skipping {vname}")
