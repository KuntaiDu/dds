# DDS

## 1. Related resources

Please check [Kuntai Du's home page](https://kuntaidu.github.io/aboutme.html) for more DDS-related resources.


## 2. Install Instructions

To run our code, please make sure that conda is installed. Then, under dds repo, run

```conda env create -f conda_environment_configuration.yml```

to install dds environment. Note that this installation assumes that you have GPU resources on your machine. If not, please edit ```tensorflow-gpu=1.14``` to ```tensorflow=1.14``` in ```conda_environment_configuration.yml```.

Now run

```conda activate dds```

to activate dds environment, and 

```cd workspace```

and run 

```wget people.cs.uchicago.edu/~kuntai/frozen_inference_graph.pb```

to download the object detection model (FasterRCNN-ResNet101).

## 3. Run our code

Under ```DDSrepo/workspace```, run

```python entrance.py```

to run DDS!

## 4. Get performance numbers

Under ```DDSrepo/workspace```, run

```python examine.py trafficcam_1 results stats```

you should see something like

```
trafficcam_1_dds_0.8_0.8_36_26_0.0_twosides_batch_15_0.5_0.3_0.01 3474KB 0.901
trafficcam_1_mpeg_0.8_26 8652KB 0.904
trafficcam_1_mpeg_0.8_36 2369KB 0.876
```

The number might vary by platform.

## 5. Some details

If you are considering building your projects based on our codebase, here are some details.

### 5.1 Run in implementation mode

Implementation means we run DDS under real network environment through HTTP post. To do so, in ```DDSrepo```, run

```FLASK_APP=backend/backend.py flask run --port=5001```

and copy the ```frozen_inference_graph.pb``` to ```DDSrepo``` to help the server find the model.

Then use another terminal, cd to ```DDSrepo/workspace```, and edit the mode to ```implementation``` and edit the hname to ```ip:5001``` (ip should be 127.0.0.1 if you run the server locally) to run DDS on implementation mode. You can also run other methods in implementation mode by changing the default value of mode to ```implementation```. 


### 5.2 Inside workspace folder

Inside workspace folder, we use a configuration file ```configuration.yml``` to control the configuration for both the client and the server. This file will be only loaded **once** inside the whole ```python entrance.py``` process. You can add new keys and values in this file. We even support caching, parameter sweeping, and some fancy functionalities in this file. Please read the comments inside this file to utilize it.


## 6. Dataset

### 6.1 Detection dataset

#### 6.1.1 Traffic camera videos and dash camera videos.

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

#### 6.1.2 Drone videos

We obtain drone videos through a public dataset called [VisDrone](https://www.aiskyeye.com). We only use 13 videos in this dataset (there are too many videos inside...).

#### 6.1.3 Face videos

We use clips from TBBT and Friends.
