
import torch
from torchvision.models.segmentation import fcn_resnet101
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from dds_utils import Region, Results
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import glob
import os
import logging
import threading
from .segmenter_rpn import RPN
import cv2
import numpy as np
matplotlib.use('Agg')

import shutil
import yaml

class Segmenter(object):

    def __init__(self):

        self.logger = logging.getLogger("semantic_segmentation")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.logger.info("loading fcn_resnet101...")
        self.model = fcn_resnet101(pretrained=True)
        self.logger.info("fcn_resnet101 loaded")

        self.model.eval().cuda()

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

    def infer(self, images, video_name, fid, requires_grad = False, requires_features = False, requires_full = False):

        x = self.transform(images)

        self.features = []
        self.requires_features = requires_features

        with torch.no_grad():

            output = self.model(x)['out']
            output = output[:, [0] + dds_env['class_ids'], :, :]
            ret = None
            if requires_full:
                ret = output
            else:
                output = torch.argmax(output, 1)
                ret = output.byte()

            print(time.time() - st)
            return ret


    def region_proposal(self, image, fid, resolution, k = dds_env['kernel_size'], topk = dds_env['num_sqrt'] * dds_env['num_sqrt']):

        def unravel_index(index, shape):
            out = []
            for dim in reversed(shape):
                out.append(index % dim)
                index = index // dim
            return tuple(reversed(out))

        def area_sum(grad):
            grad = torch.cumsum(torch.cumsum(grad, axis = 0), axis = 1)
            grad_pad = F.pad(grad, (k,k,k,k), value=-1)
            x, y = grad.shape
            grad_sum = grad[:, :] + grad_pad[0: x, 0:y] - grad_pad[k:x+k, 0:y] - grad_pad[0:x, k:y+k]
            return grad_sum

        def generate_regions(grad, results, mask):
            x, y = grad.shape

            def get_max(tensor, i, j):
                return tensor[max(0,i-k+1) : min(i+1,x), max(0,j-k+1):min(j+1,y)].max()

            def set_zero(tensor, i, j):
                tensor[max(0,i-k+1) : min(i+1,x), max(0,j-k+1):min(j+1,y)] = 0

            cnt = 0
            while cnt < topk:
                index = unravel_index(torch.argmax(area_sum(grad)), grad.shape)

                # index = unravel_index(torch.argmax(grad), grad.shape)
                # index = [min(index[0].item() + k // 2, x-1), min(index[1].item() + k // 2, y-1)]

                index = [min(index[0].item() , x-1), min(index[1].item(), y-1)]

                #if get_max(grad, index[0], index[1]) < dds_env['max_threshold']:
                #    return cnt
                if area_sum(grad)[index[0], index[1]] / (k*k) < dds_env['max_threshold']:
                    return cnt

                region = torch.zeros_like(grad).byte().cuda()
                region[index[0] - k + 1: index[0], index[1] - k + 1: index[1]] = 1
                if torch.max(region + mask) == 1:
                    results.append(Region(fid, (index[1] - k + 1) / y, (index[0] - k + 1) / x, k / y, k / x, 1.0, 'pass', resolution))
                    cnt += 1
                set_zero(grad, index[0], index[1])
            return cnt

        def clean(x, entropy, mask):
            from skimage import measure
            import numpy as np
            assert isinstance(x, np.ndarray)
            low, high = dds_env['low_runtime'], dds_env['high_runtime']
            # cast to int to prevent overflow
            x = torch.tensor(measure.label(x.astype(int))).cuda()
            nclass = torch.max(x).item()
            # cleaning
            for i in range(1, nclass + 1):
                size = torch.sum(x == i) * 1.0 / x.numel()
                # these areas are large enough. Dont expand it.
                if size> high:
                    mask[x == i] = 1
                    entropy[x == i] = 0


        with torch.no_grad():

            # prediction
            pred = self.infer(image, None, None, requires_full = True)

            # obtain entropy
            # import pdb; pdb.set_trace()
            prob = F.softmax(pred, 1)

            # import pdb; pdb.set_trace()
            entropy = 1 - torch.abs(prob[0,0:1, :, :] - prob[0, 1:2, :, :])
            # entropy = prob[0, 1:2, :, :]

            # assert (1 - torch.sum(prob, dim=1)).norm() < 1e-4
            # entropy = -torch.sum(prob * torch.log(prob), dim = 1)
            original_entropy = entropy[0,:,:].clone()



            # get label and do cleaning
            # label = torch.argmax(pred, 1).cpu().numpy()
            mask = torch.zeros_like(entropy).byte().cuda()
            # clean(label, entropy, mask)

            entropy = entropy[0, :, :]
            # entropy[entropy < dds_env['entropy_thresh']] = 0
            # mask = mask[0, :, :]

            '''
            # encourage edges
            pred = self.infer(image)
            pred = pred[0,:,:]
            pred[pred != 0] = 1
            pred = pred.cpu().data.numpy().astype(np.uint8)
            kernel = np.ones((16, 16), np.uint8)
            pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
            pred = pred - cv2.erode(pred, kernel)
            pred = torch.from_numpy(pred)
            grad[pred != 0] += 1
            '''

            grad = entropy.type(torch.DoubleTensor)
            grad_ret = grad.clone()

            results = []
            num_regions = generate_regions(grad, results, mask)

        return results, grad_ret.cpu().numpy(), grad.cpu().numpy(), original_entropy.cpu().numpy(), num_regions
