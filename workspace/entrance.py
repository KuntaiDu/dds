'''
    entrance.py - user entrance for the platform

    author: Qizheng Zhang (qizhengz@uchicago.edu)
'''

import os
import subprocess
import yaml

'''
    load_configuration - read configuration information from yaml file

    Returns: config_info - a dictionary containing info of the yaml file
'''
def load_configuration():
    with open('configuration.yml', 'r') as config:
        config_info = yaml.load(config, Loader=yaml.FullLoader) # COMMENT: does this satisfy "load once"?
    return config_info


'''
    execute_single - execute an atomic instance

    single_instance - the instance to be executed

    Returns: nothing
'''
def execute_single(single_instance):
    # unpacking
    baseline = single_instance['method']

    # to be fixed:
    # 1. ambiguity between video and vname (fixed)
    # 2. check existence of result files before execution
    # 3. gt and mpeg must be run for dds regardless of whether they are in config

    # branching based on baselines
    if baseline == 'gt':
        # unpacking
        video_name = list(single_instance['qp'].keys())[0]
        gt_qp = single_instance['qp'][video_name]
        #print(f"******************gt qp is: {gt_qp}******************")
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
        video_name = list(single_instance['qp'].keys())[0]
        mpeg_qp = single_instance['qp'][video_name]
        #print(f"******************mpeg_qp is: {mpeg_qp}******************")
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
        video_name = list(single_instance['low_qp'].keys())[0]
        original_images_dir = os.path.join(data_dir, video_name, 'src')
        low_qp = single_instance['low_qp'][video_name]
        high_qp = single_instance['high_qp'][video_name]
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
    

'''
    parameter_sweeping - recursive function for parameter sweeping

    new_instance - recursive parameter

    keys - a list of all keys of an instance

    Returns: nothing
'''
def parameter_sweeping(instances, new_instance, keys):
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

        elif (isinstance(instances[curr_key], dict) 
                and len(instances[curr_key]) > 1): 
            # more than one videos in the list of qps, need parameter sweeping
            new_instance[curr_key] = {}
            for each_video in list(instances[curr_key].keys()):
                new_instance[curr_key][each_video] = instances[curr_key][each_video]
                parameter_sweeping(instances, new_instance, keys[1:])

        elif (isinstance(instances[curr_key], dict) 
                and len(instances[curr_key]) == 1 
                and isinstance(instances[curr_key][next(iter(instances[curr_key]))], list)):
            # more than one qps in a single video, need parameter sweeping
            new_instance[curr_key] = {}
            video_name = next(iter(instances[curr_key]))
            for each_value in instances[curr_key][video_name]:
                new_instance[curr_key][video_name] = each_value
                parameter_sweeping(instances, new_instance, keys[1:])

        else: # no need for parameter sweeping
            new_instance[curr_key] = instances[curr_key]
            parameter_sweeping(instances, new_instance, keys[1:])


'''
    execute_all - execute all instances based on user's config info

    config_info - configuration information from the yaml file

    Returns: nothing
'''
def execute_all(config_info):
    all_instances = config_info['instances']

    for single_instance in all_instances:
        # to be fixed: special judgement for dds

        keys = list(single_instance.keys())
        new_instance = {} # initially empty
        parameter_sweeping(single_instance, new_instance, keys)


if __name__ == "__main__":
    # load configuration information
    config_info = load_configuration()
    data_dir = config_info['data_dir']

    # start execution
    execute_all(config_info)