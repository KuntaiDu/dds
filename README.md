# DDS Simulator

<<<<<<< HEAD
## 1. Dataset

### 1.1 Detection dataset

#### 1.1.1 Traffic camera videos and dash camera videos.

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

#### 1.1.2 Drone videos

We obtain drone videos through a public dataset called [VisDrone](https://www.aiskyeye.com). We only use 13 videos in this dataset (there are too many videos inside...).

#### 1.1.3 Face videos

We use clips from TBBT and Friends.

## 2. Install Instructions

To run our code, please make sure that conda is installed. First, run

```git checkout clean```

to enter in the clean branch of DDS. And then under the DDS repo, run

```conda env create -f conda_environment_configuration.yml```

to install dds environment. Note that this installation assumes that you have GPU resources on your machine. If not, please temporarily skip this step.

## 3. Run our code

Run

```cd workspace```

to cd into the workspace of DDS. Inside that folder, we have a configuration file ```configuration.yml``` that allows you to customize the behavior of DDS and conduct parameter sweeping on DDS easily (we put some comments there to help you understand each configuration). Then, run

```python entrance.py```

to run DDS! Note that the configuration file will be only load **once** in the above line. So you can change the configuration file freely after running the program without changing the behavior of the program.
=======
Forthcoming!
>>>>>>> 8db900f086477032b039a2794ab664521bc93e04
