"""
    entrance.py - user entrance for the platform
    author: Qizheng Zhang (qizhengz@uchicago.edu)
"""

import os
import subprocess
import yaml

def load_configuration():
    """read configuration information from yaml file

    Returns:
        dict: information of the yaml file
    """
    with open('configuration.yml', 'r') as config:
        config_info = yaml.load(config, Loader=yaml.FullLoader)
    return config_info


def execute_single(single_instance):
    """execute an atomic instance

    Args:
        single_instance (dict): the instance to be executed
    """
    # unpacking
    baseline = single_instance['method']

    # to be fixed:
    # gt and mpeg must be run for dds regardless of whether they are in config

    # branching based on baselines
    if baseline == 'gt':
        # unpacking
        video_name = single_instance['video_name']
        gt_qp = single_instance['qp']
        original_images_dir = os.path.join(data_dir, video_name, 'src')

        # skip if result file already exists
        result_file_name = f"{video_name}_gt"
        if os.path.exists(os.path.join("results", result_file_name)):
            print(f"Skipping {result_file_name}")
        # otherwise, start execution
        else:
            subprocess.run(['python', '../play_video.py', 
                            '--vid-name', f'results/{result_file_name}',
                            '--high-src', f'{original_images_dir}',
                            '--resolutions',  '1.0',
                            '--output-file', 'stats', # location of output file to be fixed
                            '--max-size', '0.3',
                            '--low-threshold', '0.3',
                            '--high-threshold', '0.8', 
                            '--enforce-iframes',
                            '--qp', f'{gt_qp}',
                            '--verbosity', 'info'])

    # assume we are running emulation
    elif baseline == 'mpeg':
        # unpacking
        video_name = single_instance['video_name']
        mpeg_qp = single_instance['qp']
        mpeg_resolution = single_instance['resolution']
        original_images_dir = os.path.join(data_dir, video_name, 'src')

        # skip if result file already exists
        result_file_name = f"{video_name}_mpeg_{mpeg_resolution}_{mpeg_qp}"
        if os.path.exists(os.path.join("results", result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            subprocess.run(['python', '../play_video.py',
                            '--vid-name', f'results/{result_file_name}',
                            '--resolutions', f'{mpeg_resolution}',
                            '--high-src', f'{original_images_dir}',
                            '--output-file', 'stats', # location of output file to be fixed
                            '--ground-truth', f'results/{video_name}_gt',
                            '--max-size', '0.3',
                            '--low-threshold', '0.3',
                            '--high-threshold', '0.3', 
                            '--enforce-iframes',
                            '--qp', f'{mpeg_qp}',
                            '--verbosity', 'info'])

    # assume we are running emulation
    elif baseline == 'dds':
        # unpacking
        video_name = single_instance['video_name']
        original_images_dir = os.path.join(data_dir, video_name, 'src')
        low_qp = single_instance['low_qp']
        high_qp = single_instance['high_qp']
        low_res = single_instance['low_res']
        high_res = single_instance['high_res']
        rpn_enlarge_ratio = single_instance['rpn_enlarge_ratio']
        batch_size = single_instance['batch_size']
        prune_score = single_instance['prune_score']
        objfilter_iou = single_instance['objfilter_iou']
        size_obj = single_instance['size_obj']

        # skip if result file already exists
        result_file_name = (f"{video_name}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}_"
                            f"{rpn_enlarge_ratio}_twosides_batch_{batch_size}_"
                            f"{prune_score}_{objfilter_iou}_{size_obj}")
        if os.path.exists(os.path.join("results", result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            subprocess.run(['python', '../play_video.py',
                            '--vid-name', f'results/{result_file_name}',
                            '--high-src', f'{original_images_dir}',
                            '--resolutions', f'{low_res}', f'{high_res}',
                            '--low-results', f'results/{video_name}_mpeg_{low_res}_{low_qp}',
                            '--output-file', 'stats', # location of output file to be fixed
                            '--ground-truth', f'results/{video_name}_gt',
                            '--max-size', '0.3',
                            '--low-threshold', '0.3',
                            '--high-threshold', '0.3', 
                            '--batch-size', f'{batch_size}',
                            '--enforce-iframes',
                            '--prune-score', f'{prune_score}',
                            '--objfilter-iou', f'{objfilter_iou}',
                            '--size-obj', f'{size_obj}',
                            '--qp', f'{low_qp}', f'{high_qp}',
                            '--verbosity', 'info', 
                            '--rpn_enlarge_ratio', f'{rpn_enlarge_ratio}'])
    

def parameter_sweeping(instances, new_instance, keys):
    """recursive function for parameter sweeping

    Args:
        instances (dict): the instance in process
        new_instance (dict): recursive parameter
        keys (list): keys of the instance in process
    """
    if keys == []: # base case
        execute_single(new_instance)
    else: # recursive step
        curr_key = keys[0]
        if (isinstance(instances[curr_key], list)): 
            # need parameter sweeping
            for each_parameter in instances[curr_key]:
                # replace the list with a single value
                new_instance[curr_key] = each_parameter
                # proceed with the other parameters in keys
                parameter_sweeping(instances, new_instance, keys[1:])
        else: # no need for parameter sweeping
            new_instance[curr_key] = instances[curr_key]
            parameter_sweeping(instances, new_instance, keys[1:])


def execute_all(config_info):
    """execute all instances based on user's config info

    Args:
        config_info (dict): configuration information from the yaml file
    """
    all_instances = config_info['instances']

    for single_instance in all_instances:
        # to be fixed: special judgement for dds

        keys = list(single_instance.keys())
        new_instance = {} # initially empty
        parameter_sweeping(single_instance, new_instance, keys)


if __name__ == "__main__":
    # load configuration information (only once)
    config_info = load_configuration()
    data_dir = config_info['data_dir']

    # start execution
    execute_all(config_info)