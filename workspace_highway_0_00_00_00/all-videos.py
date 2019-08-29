import os

data_dir = "/data/yuanx/new_dataset"
gt_qps = {
    "highway_0_00_00_00": 24,
}
mpeg_resolutions = [0.375, 0.5, 0.625, 0.75, 0.9]
mpeg_qps = {
    "highway_0_00_00_00": [26, 28, 30, 33, 35, 36, 37, 38, 39, 40],
}
dds_config = {
    "highway_0_00_00_00": [
                            (0.9,40,0.9,26,0.,15,'no_filter_combined_bboxes',0.5, 0., 0.0025),
                           ],
}
vid_names = sorted(gt_qps.keys())
video_target = 'highway_0_00_00_00'
result_target = f"results_more_{video_target}"
dataset_root = '/data/yuanx/new_dataset'

# simulate mpeg
for video in vid_names:
    # Generate gt
    if video != video_target:
        continue
    res = 1.0
    qp = gt_qps[video]
    vname = f"{video}_gt"
    if not os.path.exists(os.path.join(result_target, vname)):
        os.system(f"bash ./workspace_{video_target}/run_single.sh gt {video} {vname} {qp}")
    else:
        print(f"Skipping {vname}")
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # Run MPEG
    for res in mpeg_resolutions:
        for qp in mpeg_qps[video]:
            print(str(res) + '/' + str(qp))
            vname = f"{video}_mpeg_{res}_{qp}"
            if not os.path.exists(os.path.join(result_target, vname)):
                os.system(f"bash ./workspace_{video_target}/run_single.sh mpeg {video} {vname} {qp} {res}")
            else:
                print(f"Skipping {vname}")
    num_frames = len(os.listdir(os.path.join(dataset_root, f"{video}/src")))
    print("num_frames:", num_frames)
    # Run DDS
    # before running DDS, you must have rpn results for 2nd iteration
    # from backend.rpn_inference_func import run_rpn_inference
    # for low_res, low_qp, high_res, high_qp, rpn_enlarge_ratio, batch_size, rpn_box_source, prune_score, objfilter_iou, size_obj in dds_config[video]:
    #     low_cofig_src = os.path.join(dataset_root, f"{video}_{low_res}_{low_qp}/src")
    #     if (not os.path.exists(low_cofig_src)) or (os.path.exists(low_cofig_src) and len(os.listdir(low_cofig_src))!=num_frames):
    #         # create dataset
    #         print(f"create {video}_{low_res}_{low_qp} for RPN")
    #         scale=f"{1280*low_res}:{int(720*low_res)}"
    #         os.system(f"bash ./experiment-scripts-xin/prepare_data_for_RPN.sh {video} {low_res} {low_qp} {num_frames} {scale}" )
    #         # run RPN
    #     if not os.path.exists(f"backend/${rpn_box_source}/${video}_mpeg_${low_res}_${low_qp}"):
    #         print(f"DO {video}_{low_res}_{low_qp}_{high_res}_{high_qp} RPN")
    #         run_rpn_inference(video, 0.5, 0.3, 0.3, low_res, low_qp, high_res, high_qp, result_target)
    # #
    # import pdb; pdb.set_trace()

    for low_res, low_qp, high_res, high_qp, rpn_enlarge_ratio, batch_size, rpn_box_source, prune_score, objfilter_iou, size_obj in dds_config[video]:
        if batch_size:
            vname = f"{video}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}_{rpn_enlarge_ratio}_twosides_{rpn_box_source}_batch_{batch_size}_{prune_score}_{objfilter_iou}_{size_obj}"
            os.system(f"bash ./workspace_{video_target}/run_single.sh dds {video} {vname} {low_qp} {high_qp} {low_res} {high_res} {rpn_enlarge_ratio} {rpn_box_source} {batch_size} {prune_score} {objfilter_iou} {size_obj}")
        else:
            vname = f"{video}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}_{rpn_enlarge_ratio}_twosides_{rpn_box_source}_{prune_score}_{objfilter_iou} {size_obj}"
        # if not os.path.exists(os.path.join(result_target, vname)):
            os.system(f"bash ./workspace_{video_target}/run_single.sh dds {video} {vname} {low_qp} {high_qp} {low_res} {high_res} {rpn_enlarge_ratio} {rpn_box_source} {15} {prune_score} {objfilter_iou} {size_obj}")
        # else:
            # print(f"Skipping {vname}")
