import numpy as np
import cv2
import networkx
from networkx.algorithms.components.connected import connected_components
import os
from imageio import imread
import sys
sys.path.append("../../")
from dds_utils import calc_iou

def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current

VIDEO_WIDTH=1280
VIDEO_HEIGHT=720
PATH_TO_GT_file='gt_bbox.txt'
PATH_TO_LOW_file='low_bbox.txt'
PATH_TO_FRAMES='/data/yuanx/origin'
PATH_TO_SAVE_GT_ORIGIN='/data/yuanx/gt_origin'
PATH_TO_SAVE_GT_MERGE='/data/yuanx/gt_merge'
PATH_TO_SAVE_LOW_ORIGIN='/data/yuanx/low_origin'
PATH_TO_SAVE_LOW_MERGE='/data/yuanx/low_merge'
THRESHOLD_CONF_SCORE=0.3
THRESHOLD_MERGE=0.3


# class Region:
#     def __init__(self, fid, x, y, w, h, conf, label, resolution,
#                  origin="generic"):
#         self.fid = fid
#         self.x = x
#         self.y = y
#         self.w = w
#         self.h = h
#         self.conf = conf
#         self.label = label
#         self.resolution = resolution
#         self.origin = origin
from dds_utils import Region

def read_results_txt_dict(fname):
    """Return a dictionary with fid mapped to
       and array that contains all SingleResult objects
       from that particular frame"""
    results_dict = {}

    with open(fname, "r") as f:
        lines = f.readlines()
        f.close()

    for line in lines:
        line = line.split(",")
        fid = int(line[0])
        # x, y, w, h = [float(e) for e in line[1:5]]
        x = float(line[1])* VIDEO_WIDTH
        w = float(line[3])* VIDEO_WIDTH
        y = float(line[2])* VIDEO_HEIGHT
        h = float(line[4])* VIDEO_HEIGHT

        conf = float(line[6])
        label = line[5]
        resolution = float(line[7])
        origin = "generic"
        if len(line) > 8:
            origin = line[8]
        single_result = Region(fid, x, y, w, h, conf, label,
                               resolution, origin)

        if fid not in results_dict:
            results_dict[fid] = []

        if conf > THRESHOLD_CONF_SCORE:
            results_dict[fid].append(single_result)

    return results_dict

def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current

def pairwise_overlap_indexing_list(single_result_frame):
    pointwise = [[i] for i in range(len(single_result_frame))]
    pairwise = [[i,j] for i,x in enumerate(single_result_frame) for j,y in enumerate(single_result_frame) if i != j if filter_bbox_group(x,y)]
    return pointwise + pairwise

def simple_merge(single_result_frame, index_to_merge):
    # directly using the largest box
    bbox_large = []
    for i in index_to_merge:
        i2np = np.array([j for j in i])
        left = min(np.array(single_result_frame)[i2np], key=lambda x: x.x)
        top = min(np.array(single_result_frame)[i2np], key=lambda x: x.y)
        right = max(np.array(single_result_frame)[i2np], key=lambda x: x.x+x.w)
        bottom = max(np.array(single_result_frame)[i2np], key=lambda x: x.y+x.h)
        fid,x,y,w,h,conf,label,resolution,origin = left.fid, left.x,top.y,\
                                                    right.x + right.w -left.x, \
                                                    bottom.y + bottom.h -top.y,\
                                                    left.conf, left.label, \
                                                    left.resolution, left.origin
        single_merged_region = Region(fid,x,y,w,h,conf,label,resolution,origin)
        bbox_large.append(single_merged_region)
        # bbox_large.append([left.x,top.y,right.x + right.w -left.x,bottom.y + bottom.h -top.y])
    return bbox_large

# def draw_merged_bbox(frame, bbox_large):
#     for i in range(len(bbox_large)):
#         bbox = tuple(int(np.round(x)) for x in bbox_large[i])
#         frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 204, 0), 2)
#     return frame

def draw_bbox(frame,frame_result):
    for i in range(len(frame_result)):
        bbox = tuple([int(np.round(frame_result[i].x)), int(np.round(frame_result[i].y)), int(np.round(frame_result[i].w)), int(np.round(frame_result[i].h))])
        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 204, 0), 2)
    return frame

def overlap(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1.x, bb2.x)
    y_top = max(bb1.y, bb2.y)
    x_right = min(bb1.x+bb1.w, bb2.x+bb2.w)
    y_bottom = min(bb1.y+bb1.h, bb2.y+bb2.h)

    # no overlap
    if x_right < x_left or y_bottom < y_top:
        return False
    else:
        return True

def filter_bbox_group(bb1, bb2):
    if calc_iou(bb1, bb2) > THRESHOLD_MERGE and bb1.label == bb2.label:
        return True
    else:
        return False

def main():
    results_dict = read_results_txt_dict(PATH_TO_GT_file)
    imglist = sorted(os.listdir(PATH_TO_FRAMES))
    num_images = len(imglist)
    for frame_id, single_result_frame in enumerate(results_dict.values()):
        print(frame_id)
        # if frame_id != 86:
            # continue
        im_file = os.path.join(PATH_TO_FRAMES, imglist[frame_id])
        # im = cv2.imread(im_file)
        im_in = np.array(imread(im_file))
        im2show = im_in[:,:,::-1]

        # for merge
        overlap_pairwise_list = pairwise_overlap_indexing_list(single_result_frame)
        overlap_graph = to_graph(overlap_pairwise_list)
        grouped_bbox_idx = [c for c in sorted(connected_components(overlap_graph), key=len, reverse=True)]
        # simple merge

        # lagre box is a list of Region Object
        # Each Region x,y,w,h is NOT normalized
        large_bbox = simple_merge(single_result_frame, grouped_bbox_idx)
        im2show = draw_bbox(im2show,large_bbox)

        # for origin show
        # im2show = draw_original_bbox(im2show, single_result_frame)

        result_path = os.path.join(PATH_TO_SAVE_GT_MERGE, imglist[frame_id][:-4] + ".jpg")
        cv2.imwrite(result_path, im2show)

if __name__== "__main__":
    main()
