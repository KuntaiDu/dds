# USAGE
# python examine.py video_name results_fold stats_file gt
import sys
import os
import yaml

sys.path.append('../')

video_name = sys.argv[1]
results_direc = sys.argv[2]
stats_file = sys.argv[3]
gt_mode = 'gt'
# video_name = results_direc[11:]
# print(video_name)
dirs = os.listdir(results_direc)
from dds_utils import *


gt_confid_thresh_list = [0.3]
mpeg_confid_thresh_list = [0.5]
max_area_thresh_gt_list = [0.04]
max_area_thresh_mpeg_list = max_area_thresh_gt_list

iou_thresh = 0.3
relevant_classes = 'vehicle'

def parse_stats(stats_path):
	fname_to_size = {}
	# return total size
	with open(stats_path) as f:
		for cnt, line in enumerate(f):
			if cnt == 0:
				continue
			fields = line.split(',')
			fname = fields[0].split('/')[-1]
			total_size = int(float(fields[15])/1e3)
			fname_to_size[fname] = total_size
	return fname_to_size

def parse(file_path, gt_flag):
	fid_to_bboxes = {}
	max_fid = -1
	f = open(file_path)
	line = f.readline()
	while line:
		fields = line.split(',')
		fid = int(fields[0])
		if fid > max_fid:
			max_fid = fid
		x = float(fields[1])
		y = float(fields[2])
		w = float(fields[3])
		h = float(fields[4])
		label = fields[5]
		confid = float(fields[6])
		bbox = (x,y,w,h,label,confid)
		if fid not in fid_to_bboxes:
			fid_to_bboxes[fid] = []
		bboxes = fid_to_bboxes[fid]
		bboxes.append(bbox)
		fid_to_bboxes[fid] = bboxes
		line = f.readline()
	for fid in range(max_fid+1):
		if fid not in fid_to_bboxes:
						fid_to_bboxes[fid] = []
	return max_fid,fid_to_bboxes

def vote(max_fid, map_list, gt_confid_thresh, mpeg_confid_thresh, max_area_thresh_gt, max_area_thresh_mpeg):
	result = {}
	for fid in range(max_fid+1):
		bboxes_list = []
		for i in range(len(map_list)):
			map = map_list[i]
			bboxes = map[fid]
			bboxes = filter(bboxes, gt_flag = True, gt_confid_thresh=gt_confid_thresh, mpeg_confid_thresh=mpeg_confid_thresh, max_area_thresh_gt=max_area_thresh_gt, max_area_thresh_mpeg=max_area_thresh_mpeg)
			bboxes_list.append(bboxes)
		new_boxes = []
		for b1 in bboxes_list[0]:
			count = 1
			for i in range(len(map_list)-1):
				bboxes2 = bboxes_list[i+1]
				for b2 in bboxes2:
					if iou(b1, b2) >= 0.5:
						count += 1
						break
			if count >= 2: new_boxes.append(b1)
		result[fid] = new_boxes
	return result

def filter(bboxes, gt_flag, gt_confid_thresh, mpeg_confid_thresh, max_area_thresh_gt, max_area_thresh_mpeg):
	if gt_flag:
		confid_thresh = gt_confid_thresh
		max_area_thresh = max_area_thresh_gt

	else:
		confid_thresh = mpeg_confid_thresh
		max_area_thresh = max_area_thresh_mpeg

	result = []
	for b in bboxes:
		(x,y,w,h,label,confid) = b
		if confid >= confid_thresh and w*h <= max_area_thresh and label in relevant_classes:
			result.append(b)
	return result

def iou(b1, b2):
	(x1,y1,w1,h1,label1,confid1) = b1
	(x2,y2,w2,h2,label2,confid2) = b2
	x3 = max(x1,x2)
	y3 = max(y1,y2)
	x4 = min(x1+w1,x2+w2)
	y4 = min(y1+h1,y2+h2)
	if x3>x4 or y3>y4:
		return 0
	else:
		overlap = (x4-x3)*(y4-y3)
		return overlap/(w1*h1+w2*h2-overlap)

def eval(max_fid, map_dd, map_gt, gt_confid_thresh, mpeg_confid_thresh, max_area_thresh_gt, max_area_thresh_mpeg):
	tp_list = []
	fp_list = []
	fn_list = []
	count_list = []
	for fid in range(max_fid+1):
		bboxes_dd = map_dd[fid]
		bboxes_gt = map_gt[fid]
		bboxes_dd = filter(bboxes_dd, gt_flag = False, gt_confid_thresh=gt_confid_thresh, mpeg_confid_thresh=mpeg_confid_thresh, max_area_thresh_gt=max_area_thresh_gt, max_area_thresh_mpeg=max_area_thresh_mpeg)
		bboxes_gt = filter(bboxes_gt, gt_flag = True, gt_confid_thresh=gt_confid_thresh, mpeg_confid_thresh=mpeg_confid_thresh, max_area_thresh_gt=max_area_thresh_gt, max_area_thresh_mpeg=max_area_thresh_mpeg)
		tp = 0
		fp = 0
		fn = 0
		count = 0
		for b_dd in bboxes_dd:
			found = False
			for b_gt in bboxes_gt:
				if iou(b_dd,b_gt) >= iou_thresh:
					found = True
					break
			if found: tp += 1
			else: fp += 1
		for b_gt in bboxes_gt:
			found = False
			for b_dd in bboxes_dd:
				if iou(b_dd,b_gt) >= iou_thresh:
					found = True
					break
			if not found: fn += 1
			else: count += 1
		tp_list.append(tp)
		fp_list.append(fp)
		fn_list.append(fn)
		count_list.append(count)
	tp = sum(tp_list)
	fp = sum(fp_list)
	fn = sum(fn_list)
	count = sum(count_list)
	return tp, fp, fn, count, round(tp/(tp+fp),3), round(tp/(tp+fn),3), round((2.0*tp/(2.0*tp+fp+fn)),3)

fname_to_size = parse_stats(stats_file)
MAX_FID = -1
fid_to_bboxes_dict = {}
for file in dirs:
	# Dont parse req regions
	if "req_regions" in file or "jpg" in file or "segment_size" in file or os.path.isdir(os.path.join(results_direc,file)): continue
	max_fid, fid_to_bboxes = parse(os.path.join(results_direc,file), gt_flag = ("gt" in file))
	if max_fid > MAX_FID:
		MAX_FID = max_fid
	fid_to_bboxes_dict[file] = fid_to_bboxes

if gt_mode == 'gt':
	gt_key = video_name + "_gt"
	gt = fid_to_bboxes_dict[gt_key]
	max_f1_distance = -1
	for max_area_thresh_gt in max_area_thresh_gt_list:
		for max_area_thresh_mpeg in max_area_thresh_mpeg_list:
			for gt_confid_thresh in gt_confid_thresh_list:
				for mpeg_confid_thresh in mpeg_confid_thresh_list:
					for key in sorted(fid_to_bboxes_dict):
						if key not in fname_to_size:
							continue
						if 'gt' in key:
							continue
						print(key, fname_to_size[key], end = 'KB ')

						tp, fp, fn, count, pr, recall, f1 = eval(MAX_FID, fid_to_bboxes_dict[key], gt, gt_confid_thresh, mpeg_confid_thresh, max_area_thresh_gt, max_area_thresh_mpeg)

						# print(f1, tp, fp, fn, count)
						print(f1)
