import os


data_dir = "../new_dataset"
gt_qps = {
    "drivers_0_00_00_00": 21,
    "flyover_1_00_00_00": 26,
    "flyover_2_00_00_00": 28,
    "highway_0_00_00_00": 24,
    "highway_0_24_00_00": 25,
    "highway_1_24_00_00": 28,
    "interchange_0_00_00_00": 25,
    "interchange_1_00_00_00": 25,
    "roadsmall_0_00_00_00": 26
}
mpeg_resolutions = [0.15, 0.2, 0.25, 0.375, 0.5, 0.625, 0.75]
dds_config = {
    "drivers_0_00_00_00": ([(0.5, 0.2), (0.375, 0.2)], [22, 28, 36]),
    "flyover_1_00_00_00": ([(0.5, 0.2), (0.375, 0.2)], [28, 36]),
    "flyover_2_00_00_00": ([(0.5, 0.2), (0.375, 0.2)], [30, 34, 38]),
    "highway_0_00_00_00": ([(0.5, 0.2), (0.375, 0.2)], [28, 34, 38]),
    "highway_0_24_00_00": ([(0.5, 0.2), (0.375, 0.2)], [27, 33]),
    "highway_1_24_00_00": ([(0.5, 0.2), (0.375, 0.2)], [30, 38]),
    "interchange_0_00_00_00": ([(0.5, 0.2), (0.375, 0.2)], [35, 29]),
    "interchange_1_00_00_00": ([(0.5, 0.2), (0.375, 0.2)], [29, 37]),
    "roadsmall_0_00_00_00": ([(0.5, 0.2), (0.375, 0.2)], [36, 30])
}
vid_names = sorted(gt_qps.keys())

for video in vid_names:
    # Generate gt
    res = 1.0
    qp = gt_qps[video]
    vname = f"{video}_gt"
    os.system(f"bash run_single.sh gt {video} {vname} {qp}")

    # Run MPEG
    for res in mpeg_resolutions:
        _, qps = dds_config[video]
        for qp in qps:
            vname = f"{video}_mpeg_{res}_{qp}"
            os.system(f"bash run_single.sh mpeg {video} {vname} {qp} {res}")

    # Run DDS
    res_config, qps = dds_config[video]
    for high, low in res_config:
        for qp in qps:
            vname = f"{video}_dds_{low}_{high}_{qp}"
            os.system(f"bash run_single.sh dds {video} {vname} {qp} {low} {high}")
