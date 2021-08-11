
import subprocess

name_list_trafficcam = ['trafficcam_%d' % (i+1) for i in range(7)]
name_list_dashcam = ['dashcam_%d' % (i+1) for i in range(9)]
# quick fix
name_list_drone = ['drone_%d' % i for i in [14,15,16,17,18,19,20,21,22,23,24,29,30]]
name_list = name_list_trafficcam + name_list_dashcam + name_list_drone

name_list = name_list_trafficcam

for name in name_list:
    print('Processing %s......' % name)
    # python play_video.py --vid-name results_drone_28/drone_28_vigil --resolutions 1 --qp 20 --high-src /data/kuntai/new_dataset/drone_28/src/ --baseline vigil --output-file stats_drone_28


    subprocess.run([
        'python',
        'play_video.py',
        '--vid-name',
        f'results_{name}/{name}_vigil',
        '--resolutions',
        '1.0',
        '--qp',
        '24',
        '--high-src',
        f'/data2/kuntai-gpuc/kuntai/new_dataset/{name}/src/',
        '--baseline',
        'vigil',
        '--output-file',
        f'stats_{name}'
        ])
    #subprocess.run([
    #    'python',
    #    'play_video.py',
    #    '--vid-name',
    #    f'results_{name}/{name}_glimpse',
    #    '--resolutions',
    #    '1.0',
    #    '--qp',
    #    '24',
    #    '--high-src',
    #    f'/data2/kuntai-gpuc/kuntai/new_dataset/{name}/src/',
    #    '--baseline',
    #    'glimpse',
    #    '--output-file',
    #    f'stats_{name}'
    #    ])
