
import os
import argparse
import sys
import yaml

# append the root of dds to find dds_config_generator
with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())
sys.path.append(dds_env['root'])

import dds_config_generator

parser = argparse.ArgumentParser()
parser.add_argument('CODE', type=int)
args = parser.parse_args()
CODE = args.CODE

names, mpeg_configs, dds_config, gt_config = dds_config_generator.load()

data_dir = dds_env['dataset']
vid_names = names
dataset_root = dds_env['dataset']


#not_list = ['dashcam_%d' % (i+1) for i in range(8)] + ['trafficcam_%d' % (i+1) for i in range(6)]
not_list = []

# simulate mpeg
for video_target in vid_names:
    result_target = f"results_{video_target}"
    # Generate gt
    video = video_target
    res, qp = gt_config[video]
    vname = f'{video}_gt'
    if not os.path.exists(os.path.join(result_target, vname)):
        os.system(f"python ./workspace/run_single.py gt {video} {vname} {qp} {res}")

    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # Run MPEG
    if CODE==0:
        for res, qp in mpeg_configs[video]:
            print(str(res) + '/' + str(qp))
            vname = f"{video}_mpeg_{res}_{qp}"
            if True:
                os.system(f"python ./workspace/run_single.py mpeg {video} {vname} {qp} {res}")
            else:
                print(f"Skipping {vname}")
    num_frames = len(os.listdir(os.path.join(dataset_root, f"{video}/src")))
    print("num_frames:", num_frames)
    # Run DDS
    # before running DDS, you must have rpn results for 2nd iteration
    if CODE==1:
        from backend.rpn_inference_func import run_rpn_inference
        for low_res, low_qp, high_res, high_qp, rpn_enlarge_ratio, batch_size, rpn_box_source, prune_score, objfilter_iou, size_obj in dds_config[video]:
            low_cofig_src = os.path.join(dataset_root, f"{video}_{low_res}_{low_qp}/src")
            if (not os.path.exists(low_cofig_src)) or (os.path.exists(low_cofig_src) and len(os.listdir(low_cofig_src))!=num_frames):
             # create dataset
                print(f"create {video}_{low_res}_{low_qp} for RPN")
                scale=f"{1280*low_res}:{int(720*low_res)}"
                os.system(f"python ./experiment-scripts-xin/prepare_data_for_RPN.py {video} {low_res} {low_qp} {num_frames} {scale}" )
             # run RPN
            if not os.path.exists(f"backend/${rpn_box_source}/${video}_mpeg_${low_res}_${low_qp}"):
                print(f"DO {video}_{low_res}_{low_qp}_{high_res}_{high_qp} RPN")
                run_rpn_inference(video, 0.5, 0.3, 0.3, low_res, low_qp, high_res, high_qp, result_target)
     #

    if CODE==2:
        for low_res, low_qp, high_res, high_qp, rpn_enlarge_ratio, batch_size, rpn_box_source, prune_score, objfilter_iou, size_obj in dds_config[video]:
            if batch_size:
                vname = f"{video}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}_{rpn_enlarge_ratio}_twosides_{rpn_box_source}_batch_{batch_size}_{prune_score}_{objfilter_iou}_{size_obj}"
                os.system(f"python ./workspace/run_single.py dds {video} {vname} {low_qp} {high_qp} {low_res} {high_res} {rpn_enlarge_ratio} {rpn_box_source} {batch_size} {prune_score} {objfilter_iou} {size_obj}")
            else:
                vname = f"{video}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}_{rpn_enlarge_ratio}_twosides_{rpn_box_source}_{prune_score}_{objfilter_iou} {size_obj}"
        # if not os.path.exists(os.path.join(result_target, vname)):
                os.system(f"python ./workspace/run_single.py dds {video} {vname} {low_qp} {high_qp} {low_res} {high_res} {rpn_enlarge_ratio} {rpn_box_source} {15} {prune_score} {objfilter_iou} {size_obj}")
        # else:
            # print(f"Skipping {vname}")

    # only for debug
