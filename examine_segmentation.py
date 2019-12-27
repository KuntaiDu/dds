# USAGE
# python examine.py video_name results_fold stats_file gt
import sys
import os
import yaml
import pickle
import numpy
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np


with open('dds_env.yaml', 'r') as f:
	dds_env = yaml.load(f.read())

video_name = sys.argv[1]
results_direc = sys.argv[2]
stats_file = sys.argv[3]
gt_mode = sys.argv[4]
# video_name = results_direc[11:]
print(video_name)
dirs = os.listdir(results_direc)

def parse_stats(stats_path):
	fname_to_size = {}
	# return total size
	with open(stats_path) as f:
		for cnt, line in enumerate(f):
			if cnt == 0:
				continue
			fields = line.split(',')
			fname = fields[0].split('/')[-1]
			total_size = int(float(fields[15]))
			fname_to_size[fname] = total_size
	return fname_to_size


fname_to_size = parse_stats(stats_file)
results_direc = Path(results_direc)

for name in dirs:
	# skip unwanted files
	if "req_regions" in name or "jpg" in name or "segment_size" in name or (results_direc / name).is_dir():
		continue

	print(f'Processing {name}', file = sys.stderr)

	with open(f'{results_direc / name}', 'rb') as f:
		x = pickle.load(f)

	with open(f'{results_direc / video_name}_gt', 'rb') as f:
		gt = pickle.load(f)

	assert x.keys() == gt.keys()

	F1 = []
	for key in x.keys():
		if isinstance(x[key], numpy.ndarray):
			x[key] = torch.from_numpy(x[key])
		if isinstance(gt[key], numpy.ndarray):
			gt[key] = torch.from_numpy(gt[key])
		#import pdb; pdb.set_trace()

		x_ind = x[key]
		gt_ind = gt[key]
		assert gt_ind.shape == x_ind.shape
		'''
		print(key2label[gt_ind_i[0]])
		print(_[0])
		print(key2label[gt_ind_i[1]])
		print(_[1])
		print(key2label[gt_ind_i[2]])
		print(_[2])
		print(key2label[gt_ind_i[3]])
		print(_[3])
		print(key2label[gt_ind_i[4]])
		print(_[4])
		'''

		# skip background class
		background = torch.zeros_like(gt_ind)

		# filter out all background pixels
		correct_pixels = torch.where( (gt_ind == x_ind) & ((gt_ind != background) | (x_ind != background)) , torch.ones_like(gt_ind), torch.zeros_like(gt_ind))
		total_pixels = torch.where(((gt_ind != background) | (x_ind != background)), torch.ones_like(gt_ind), torch.zeros_like(gt_ind))

		ncorrect = len(correct_pixels.nonzero())
		ntotal = len(total_pixels.nonzero())
		F1.append(ncorrect/ntotal)


	print(f'{name} {fname_to_size[name]} {np.mean(F1)} 0')
