
import sys
import os
from pathlib import Path

import yaml

args = sys.argv

with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())


dataset = Path(dds_env['dataset'])

mode = args[1]
video = args[2]

output_file = Path(f'stats_{video}')
results_dir = Path(f'results_{video}')
original_images = dataset/video/'src'

os.system(f"mkdir -p f{results_dir}")

if mode == 'gt':

    vname = args[3]
    qp = args[4]
    res = args[5]

    os.system(f"python play_video.py --vid-name {results_dir/vname} --high-src ${original_images} "\
    f"--resolutions 1.0 --output-file {output_file} --max-size 0.3 --low-threshold 0.3 "\
    f"--high-threshold 0.8 --enforce-iframes --qp {qp} --verbosity info")

elif mode == 'mpeg':

    vname = args[3]
    qp = args[4]
    res = args[5]

    os.system(f"python play_video.py --vid-name {results_dir/vname} --resolutions ${res} "\
    f"--high-src {original_images} --output-file {output_file} "\
    f"--ground-truth {results_dir / (video + '_gt')} --max-size 0.3 --low-threshold 0.3 "\
    f"--high-threshold 0.3 --enforce-iframes --qp ${qp} --verbosity info")

elif mode == 'dds':

    _, _, vid_name, vname, low_qp, high_qp, low, high, rpn_enlarge_ratio,\
        rpn_box_source, batch_size, prune_score, objfilter_iou, size_obj = args
    low_results = Path('backend')/rpn_box_source/f'{video}_mpeg_{low}_{low_qp}'

    os.system(f"python play_video.py --vid-name {results_dir/vname} --high-src {original_images} "\
    f"--resolutions {low} {high} --low-results {low_results} --output-file {output_file} "\
    f"--ground-truth {results_dir/(video + '_gt')} --max-size 0.3 "\
    f"--low-threshold 0.3 --high-threshold 0.3 --enforce-iframes --qp {low_qp} {high_qp} "\
    f"--verbosity info --rpn_enlarge_ratio {rpn_enlarge_ratio} "\
    f"--batch-size {batch_size} --prune-score {prune_score} --objfilter-iou {objfilter_iou} "\
    f" --size-obj {size_obj}")