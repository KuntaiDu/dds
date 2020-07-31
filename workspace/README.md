# Integrated Video Analytics Platform

## Get Started
Before you get started, make sure that you have installed the following libraries:<br/>
munch<br/>

(To be updated)

### Set up the configuration file
configuration.yaml is the configuration file that you need to set up and from
which the main program would be reading. If you are unfamiliar with the format
of yaml files, use generate_yaml.py to generate one.

### Run the program

## Naming Convention (for developers)
### General variable names
- data_dir: path of the directory that contain all the video datasets
- method: baseline/method to be executed
- qp: quantization parameter<br/>
  gt_qp for groundtruth, mpeg_qp for AWStream, low_qp and high_qp for DDS
- resolution: resolution of the videos<br/>
  mpeg_resolution for AWStream, low_res and high_res for DDS
- video_name: name of the video dataset to be streamed

### DDS-related variable names
batch_size: number of frames to be processed at one time
objfilter_iou
prune_score
rpn_enlarge_ratio
size_obj

## To-dos (for developers)
- Interface for inference results
- Interface for different applications
- Then, we are able to add more applications and baselines