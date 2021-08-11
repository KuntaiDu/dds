
from dds_utils import *
from backend.merger import *
from pathlib import Path

video_name = 'trafficcam_1'

for res, qp in [[0.25, 1], [0.25, 11], [0.25,21]]:
    directory = Path(f'./results_{video_name}')
    filename = f'{video_name}_mpeg_{res}_{qp}_cloudseg'

    absolute = directory / filename

    dic = read_results_txt_dict(str(absolute))
    results = merge_boxes_in_results(dic, 0.3, 0.3)
    results.write(str(absolute) + '_merge')