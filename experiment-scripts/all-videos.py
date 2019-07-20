import os


data_dir = "../new_dataset"
gt_qps = {
    "manhattan_021_00_00_00": 34,
    "flyover_4_00_00_00": 32,
    "flyover_2_00_00_00": 30,
    "highway_1_00_00_00": 31,
    "highway_1_45_00_00": 31,
    "ladash_002_00_00_00": 34,
    "ladash_000_00_00_00": 33,
    "street_0_00_00_07": 30,
    "flyover_1_00_00_00": 31,
    "interchange_0_00_00_1": 30,
    "interchange_1_00_00_00": 29,
    "highway_1_24_00_00": 31,
    "interchange_0_00_00_00": 30,
    "highway_1_47_00_00": 31
}
mpeg_resolutions = [0.15, 0.2, 0.25, 0.375, 0.5, 0.625, 0.75]
mpeg_qps = {
    "manhattan_021_00_00_00": [35, 36, 37, 38, 39, 40],
    "flyover_4_00_00_00": [34, 35, 36, 37, 38, 39, 40],
    "flyover_2_00_00_00": [33, 35, 36, 37, 38, 39, 40],
    "highway_1_00_00_00": [33, 35, 36, 37, 38, 39, 40],
    "highway_1_45_00_00": [33, 35, 36, 37, 38, 39, 40],
    "ladash_002_00_00_00": [35, 36, 37, 38, 39, 40],
    "ladash_000_00_00_00": [35, 36, 37, 38, 39, 40],
    "street_0_00_00_07": [33, 35, 36, 37, 38, 39, 40],
    "flyover_1_00_00_00": [33, 35, 36, 37, 38, 39, 40],
    "interchange_0_00_00_1": [33, 35, 36, 37, 38, 39, 40],
    "interchange_1_00_00_00": [32, 35, 36, 37, 38, 39, 40],
    "highway_1_24_00_00": [33, 35, 36, 37, 38, 39, 40],
    "interchange_0_00_00_00": [33, 35, 36, 37, 38, 39, 40],
    "highway_1_47_00_00": [33, 35, 36, 37, 38, 39, 40]
}
dds_config = {
    "flyover_4_00_00_00": [(0.25, 0.5, 40, 40), (0.25, 0.5, 39, 40)],
    "highway_1_45_00_00": [(0.25, 0.5, 40, 40)],
    "street_0_00_00_07": [(0.375, 0.625, 40, 40)],
    "interchange_0_00_00_1": [(0.25, 0.5, 38, 39)],
    "highway_1_47_00_00": [(0.25, 0.5, 40, 40), (0.2, 0.5, 40, 40)],
    "ladash_002_00_00_00": [(0.25, 0.5, 40, 40), (0.2, 0.375, 39, 40)],
    "ladash_000_00_00_00": [(0.25, 0.5, 40, 40), (0.25, 0.375, 40, 38), (0.2, 0.375, 39, 40)]
}
vid_names = sorted(dds_config.keys())

for video in vid_names:
    # # Generate gt
    # res = 1.0
    # qp = gt_qps[video]
    # vname = f"{video}_gt"
    # if not os.path.exists(os.path.join("results", vname)):
    #     os.system(f"bash run_single.sh gt {video} {vname} {qp}")
    # else:
    #     print(f"Skipping {vname}")

    # # Run MPEG
    # for res in mpeg_resolutions:
    #     for qp in mpeg_qps[video]:
    #         vname = f"{video}_mpeg_{res}_{qp}"
    #         if not os.path.exists(os.path.join("results", vname)):
    #             os.system(f"bash run_single.sh mpeg {video} {vname} {qp} {res}")
    #         else:
    #             print(f"Skipping {vname}")

    # Run DDS
    for low_res, high_res, low_qp, high_qp in dds_config[video]:
        vname = f"{video}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}"
        if not os.path.exists(os.path.join("results", vname)):
            os.system(f"bash run_single.sh dds {video} {vname} {low_qp} {high_qp} {low_res} {high_res}")
        else:
            print(f"Skipping {vname}")
