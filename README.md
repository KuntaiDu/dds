# dds

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
