import yaml

configuration = {
    'data_dir': '/tank/qizheng/new_dataset',
    'mode': 'emulation',
    'instances': [
        {
            'method': 'gt',
            'video_name': 'trafficcam_1',
            'qp': 34
            },
        {
            'method': 'mpeg',
            'video_name': 'trafficcam_1',
            'resolution': [0.375, 0.5, 0.625, 0.7, 0.75, 0.8, 0.9],
            'qp': [35, 36, 40]
            },
        {
            'method': 'dds',
            'video_name': 'trafficcam_1',
            'low_res': 0.9,
            'low_qp': 40,
            'high_res': 0.9,
            'high_qp': 26,
            'rpn_enlarge_ratio': 0.,
            'batch_size': 15,
            'prune_score': 0.5,
            'objfilter_iou': 0.,
            'size_obj': 0.0025
            }
    ]
}

with open('configuration.yml', 'w') as config:
    documents = yaml.dump(configuration, config)
    print(documents)