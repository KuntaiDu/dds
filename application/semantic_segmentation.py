
import torch
from torchvision.models.segmentation import fcn_resnet101
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import yaml
import glob
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

from .application import Application

import shutil
import yaml

from results.regions import (Regions, Region)
from results.masks import Masks



class Semantic_Segmentation(Application):

    def __init__(self, server):

        self.server = server
        self.logger = logging.getLogger("semantic_segmentation")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        # image normalization
        self.im2tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # reverse the normalization process, for visualization purpose
        self.unnormalize = T.Compose([
            T.Normalize(
                mean = [-2.118, -2.036, -1.804],
                std = [4.367, 4.464, 4.444])
        ])

    def transform(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.cat([self.im2tensor(i)[None,:,:,:].cuda() for i in images], dim=0)
        return images

    def create_empty_results(self):
        return Masks()

    def get_deserializer(self):
        return lambda x: Masks(x)

    def run_inference(self, model, images_direc, resolution, fnames=None, images=None, require_full = False, config = None):

        if fnames is None:
            fnames = sorted(ost.listdir(images_direc))

        if model == None:
            if not hasattr(self, 'model'):
                self.logger.warning("Segmentation model not found, use pytorch official model fcn_resnet101 instead.")
                self.model = fcn_resnet101(pretrained=True)
                self.logger.info("fcn_resnet101 loaded")
            model = self.model 

        self.model.eval().cuda()

        self.logger.info(f"Running semantic segmentation on {len(fnames)} frames")

        results = {}

        for fname in fnames:

            #import pdb; pdb.set_trace()

            if "png" not in fname:
                continue
            
            fid = int(fname.split(".")[0])

            image = plt.imread(Path(images_direc) / fname)
            normalized_image = self.transform([image])

            # inference, gradient not needed
            with torch.no_grad():
                output = self.model(normalized_image)['out']
                # filter out classes of interest
                output = output[:, [0] + [config['class_id']]]
                ret = None
                if require_full:
                    # require full inference results, return the probabilities
                    ret = output
                else:
                    # require inference results, return the label of each pixel
                    output = torch.argmax(output, 1)
                    # use byte to save space
                    ret = output.byte().cpu()

            results[fid] = ret

        if not require_full:
            results = Masks(results)

        return {
            "results": results
        }



    def run_inference_with_feedback(self, start_fid, end_fid, segmenter, images_direc, fnames, config):

        results = self.run_inference(segmenter, images_direc, config.low_resolution, fnames, config=config, require_full=True)['results']

        inference_results = Masks()
        feedback_regions = Regions()

        for fid in results.keys():
            result = results[fid]
            inference_results.masks[fid] = torch.argmax(result, 1).byte().cpu()
            for region in self. feedback_region_proposal(fid, result, config):
                print(region.toJSON())
                feedback_regions.append(region)
            # import pdb; pdb.set_trace()

        return {
            'inference_results': inference_results.toJSON(),
            'feedback_regions': feedback_regions.toJSON()
        }

    def postprocess_results(self):
        return lambda x, y: x
            

        

    def feedback_region_proposal(self, fid, pred, config):

        k = config['kernel_size']
        topk = config['num_sqrt'] * config['num_sqrt']
        resolution = config['low_resolution']

        if fid % 10 == 15:
            self.logger.info(f'Using kernel size {k} and propose {topk} regions.')

        def unravel_index(index, shape):
            # To get the coordinate of the maximum element
            out = []
            for dim in reversed(shape):
                out.append(index % dim)
                index = index // dim
            return tuple(reversed(out))

        def area_sum(grad):
            # get the sum of objectness score of k*k rectangles.
            grad = torch.cumsum(torch.cumsum(grad, axis = 0), axis = 1)
            # pad with -1 to let the bounding box lie inside the image
            grad_pad = F.pad(grad, (k,k,k,k), value=-1)
            x, y = grad.shape
            grad_sum = grad[:, :] + grad_pad[0: x, 0:y] - grad_pad[k:x+k, 0:y] - grad_pad[0:x, k:y+k]
            return grad_sum

        def generate_regions(objectness, results, mask):
            x, y = objectness.shape

            def get_max(tensor, i, j):
                return tensor[max(0,i-k+1) : min(i+1,x), max(0,j-k+1):min(j+1,y)].max()

            def set_zero(tensor, i, j):
                tensor[max(0,i-k+1) : min(i+1,x), max(0,j-k+1):min(j+1,y)] = 0

            cnt = 0
            while cnt < topk:
                index = unravel_index(torch.argmax(area_sum(objectness)), objectness.shape)

                # this part is used for segmenting motorcycle.
                # index = unravel_index(torch.argmax(objectness), objectness.shape)
                # index = [min(index[0].item() + k // 2, x-1), min(index[1].item() + k // 2, y-1)]

                index = [min(index[0].item() , x-1), min(index[1].item(), y-1)]

                region = torch.zeros_like(objectness).byte().cuda()

                region[index[0] - k + 1: index[0], index[1] - k + 1: index[1]] = 1
                if torch.max(region + mask) == 1:
                    results.append(Region(fid, (index[1] - k + 1) / y, (index[0] - k + 1) / x, k / y, k / x, 1.0, 'pass', resolution))
                    cnt += 1
                set_zero(objectness, index[0], index[1])

            return cnt

        def clean(x, objectness, mask):
            # filter out those objects that are too large
            assert isinstance(x, np.ndarray)
            high = config['high_obj_size_semantic_segmentation']
            # cast to int to prevent overflow
            x = torch.tensor(measure.label(x.astype(int))).cuda()
            nclass = torch.max(x).item()
            # perform the filter
            for i in range(1, nclass + 1):
                size = torch.sum(x == i) * 1.0 / x.numel()
                if size> high:
                    mask[x == i] = 1
                    objectness[x == i] = 0

        with torch.no_grad():

            # softmax to get probability
            prob = F.softmax(pred, 1)

            # obtain objectness score
            objectness = 1 - torch.abs(prob[0,0:1, :, :] - prob[0, 1:2, :, :])

            # clean those large objects
            label = torch.argmax(pred, 1).cpu().numpy()
            mask = torch.zeros_like(objectness).byte().cuda()
            clean(label, objectness, mask)

            # obtain 2-D-shaped objectness
            objectness = objectness[0, :, :].type(torch.DoubleTensor)

            regions = []
            num_regions = generate_regions(objectness, regions, mask)

        return regions