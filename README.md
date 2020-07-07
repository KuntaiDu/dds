# dds

## Dataset

### 1. Detection dataset

#### 1.1. Traffic camera videos and dash camera videos.

We search some keywords through youtube in the anonymous mode of Chrome. The top-ranked search results, corresponding URLS are listed below. We filter out some of these videos.

| Keyword                | Source  | Type       | URL                                           | Why we filter it out |
| ---------------------- | ------- | ---------- | --------------------------------------------- | -------------------- |
|                        |         |            |                                               |                      |
| city drive around      | youtube | dashcam    | <https://www.youtube.com/watch?v=7HaJArMDKgI> |                      |
| city drive around      | youtube | dashcam    | <https://www.youtube.com/watch?v=kOMWAnxKq58> |                      |
| city drive around      | youtube | dashcam    | <https://www.youtube.com/watch?v=RTLwaQFtXbE> | night                |
| city drive around      | youtube | trafficcam | <https://www.youtube.com/watch?v=g_4RT0We1F8> | nearly no object     |
| city drive around      | youtube | dashcam    | <https://www.youtube.com/watch?v=Cw0d-nqSNE8> |                      |
| city drive around      | youtube | dashcam    | <https://www.youtube.com/watch?v=fkps18H3SXY> |                      |
| city drive around      | youtube | trafficcam | <https://www.youtube.com/watch?v=Ujyu8foke60> | night                |
| city drive around      | youtube | dashcam    | <https://www.youtube.com/watch?v=7o5PYCeEo2I> |                      |
| city drive around      | youtube |            | <https://www.youtube.com/watch?v=lTvYjERVAnY> | night                |
| city drive around      | youtube | dashcam    | <https://www.youtube.com/watch?v=6tyFAtgy4JA> |                      |
| city drive around      | youtube | dashcam    | <https://www.youtube.com/watch?v=n1xkO0_lSU0> | night                |
| city drive around      | youtube | dashcam    | <https://www.youtube.com/watch?v=LF22Ybb_pyQ> |                      |
| city drive around      | youtube | dashcam    | <https://www.youtube.com/watch?v=y1OCipyZefA> |                      |
| city drive around      | youtube | dashcam    | <https://www.youtube.com/watch?v=2LXwr2bRNic> |                      |
|                        |         |            |                                               |                      |
|                        |         |            |                                               |                      |
| highway traffic camera | youtube | trafficcam | <https://www.youtube.com/watch?v=MNn9qKG2UFI> |                      |
| highway traffic camera | youtube | trafficcam | <https://www.youtube.com/watch?v=PJ5xXXcfuTc> |                      |
| highway traffic camera | youtube | trafficcam | <https://www.youtube.com/watch?v=y3NOhpkoR-w> |                      |
| highway traffic camera | youtube | dashcam    | <https://www.youtube.com/watch?v=hxyhulJYz5I> |                      |
| highway traffic camera | youtube | trafficcam | <https://www.youtube.com/watch?v=5_XSYlAfJZM> |                      |
| highway traffic camera | youtube | trafficcam | <https://www.youtube.com/watch?v=b46xvHwxpcY> | low resolution       |
| highway traffic camera | youtube |            | <https://www.youtube.com/watch?v=4koxy_7uqcg> | not a real video     |
| highway traffic camera | youtube |            | <https://www.youtube.com/watch?v=jjlBnrzSGjc> |                      |
| highway traffic camera | youtube | trafficcam | <https://www.youtube.com/watch?v=fxec0tHMkk4> |                      |
| highway traffic camera | youtube | dashcam    | <https://www.youtube.com/watch?v=jQcuhLqebPk> |                      |
| highway traffic camera | youtube |            | <https://www.youtube.com/watch?v=8XoTvbqsT68> | not a real video     |
| highway traffic camera | youtube |            | <https://www.youtube.com/watch?v=6VsYw7NcQLo> | not a real video     |
| highway traffic camera | youtube |            | <https://www.youtube.com/watch?v=HLiAJ0gW0kk> | need 18+             |
| highway traffic camera | youtube | trafficcam | <https://www.youtube.com/watch?v=1EiC9bvVGnk> |                      |
| highway traffic camera | youtube | trafficcam | <https://www.youtube.com/watch?v=WxgtahHmhiw> |                      |
| highway traffic camera | youtube | trafficcam | <https://www.youtube.com/watch?v=PmrSOPMkfAo> | night                |
| highway traffic camera | youtube |            | <https://www.youtube.com/watch?v=X7qGtl9lW2A> | not a real video     |
| highway traffic camera | youtube |            | <https://www.youtube.com/watch?v=pcpL9rAhRVA> | night                |
| highway traffic camera | youtube |            | <https://www.youtube.com/watch?v=w6gs10P2e1k> | not a real video     |

After downloading these videos, we get 7 traffic camera videos and 9 dash camera videos.

#### 1.2. Drone videos

We obtain drone videos through a public dataset called [aiskyeye](https://www.aiskyeye.com). We only use 13 videos in this dataset.

#### 1.3. Face videos
We use clips from TBBT and Friends.

## Install Instructions

First of all, please ensure that conda is installed. Then create a new environment and run:

``` conda install tensorflow-gpu=1.14 ```

to install a 1.14.0 version of tensorflow-gpu.

Also install ```networkx```:

``` conda install networkx ```

and ```yaml```:

``` conda install -c anaconda pyyaml ```


Then use

`` pip install opencv-contrib-python ``

to install opencv. We use ```pip``` instead of ```conda``` to make sure that we use the ```ffmpeg``` under ```/usr/bin/```.

Then we download the pre-trained model. Use

``` wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz ```

to download a tar ball. Then unzip it and copy the ```frozen_inference_graph.pb``` to ```PATH/TO/DDS/model/```

After that, please edit the ```dds_env.yaml```.

## Run our code

We first create some new folders. At ```PATH/TO/DDS```, run

```
mkdir backend/no_filter_combined_bboxes
```

and

```
mkdir backend/no_filter_combined_merged_bboxes
```

to create temporal folders for dds to save its region proposal results.

At ```PATH/TO/DDS```, run

``` python workspace/all-videos.py 0 ```

and run

``` python workspace/all-videos.py 1 ```

and run

``` python workspace/all-videos.py 2 ```

to run our code.

233test
