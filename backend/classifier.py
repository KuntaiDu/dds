
import torch
from torchvision.models import vgg19
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from dds_utils import Region, Results
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import glob
import os
import logging
import threading
from .classifier_rpn import RPN

import yaml
with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())

class Classifier(object):

    # Ensure Classifier class is a singleton
    __instance = None
    __lock = threading.Lock()
    def __new__(cls):

        if Classifier.__instance is None:
            with Classifier.__lock:
                if Classifier.__instance is None:
                    Classifier.__instance = super(Classifier, cls).__new__(cls)
        return Classifier.__instance


    def __init__(self):

        self.logger = logging.getLogger("classifier")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.logger.info("loading vgg19")
        self.model = vgg19(pretrained=True)
        self.model.eval().cuda()
        self.requires_features = False
        self.features = []
        self.logger.info("vgg19 loaded to gpu")

        self.im2tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.logger.info("loading rpn")
        self.rpn = RPN(self)
        self.logger.info("rpn loaded to gpu")


    def transform(self, images):
        if not isinstance(images, torch.Tensor):
            return torch.cat([self.im2tensor(i)[None,:,:,:].cuda() for i in images], dim=0)
        else:
            return images

    def infer(self, images, requires_grad = False, requires_features = False):

        x = self.transform(images)
        self.features = []
        self.requires_features = requires_features

        if not requires_grad:
            func = torch.no_grad
        else:
            func = torch.enable_grad
        with func():
            for module in self.model.features:
                # import pdb; pdb.set_trace()
                if self.requires_features and isinstance(module, torch.nn.MaxPool2d):
                    self.features.append(x)
                x = module(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            x = F.softmax(x, dim = 1)[0]
        return x


    def region_proposal(self, image, fid, resolution, k = dds_env['kernel_size'], topk = dds_env['num_sqrt'] * dds_env['num_sqrt']):

        def unravel_index(index, shape):
            out = []
            for dim in reversed(shape):
                out.append(index % dim)
                index = index // dim
            return tuple(reversed(out))

        def generate_regions(grad, results):
            x, y = grad.shape

            def set_zero(tensor, i, j):
                tensor[max(0, i+1-k) : min(x, i+k), max(0, j+1-k):min(y, j+k)] = 0

            for i in range(topk):
                index = unravel_index(torch.argmax(grad), grad.shape)
                index = [index[0].item(), index[1].item()]
                results.append(Region(fid, (index[1] - k + 1) / y, (index[0] - k + 1) / x, k / y, k / x, 1.0, 'pass', resolution))
                set_zero(grad, index[0], index[1])

        # assert k % 2 == 1

        # assert len(image) == 1


        with torch.no_grad():
            self.infer(image, requires_features = True)
            grad = self.rpn(self.features)
            grad = grad[0,0,:,:]
            grad = torch.abs(grad).type(torch.DoubleTensor)
            grad = grad.type(torch.DoubleTensor)
            grad = torch.cumsum(torch.cumsum(grad, axis = 0), axis = 1)
            grad_pad = F.pad(grad, (k,k,k,k))
            x, y = grad.shape
            grad_sum = grad[:, :] + grad_pad[0: x, 0:y] - grad_pad[k:x+k, 0:y] - grad_pad[0:x, k:y+k]

            results = []
            generate_regions(grad_sum, results)

        return results


def run_rpn_inference(video, _,__,___, low_scale, low_qp, high_scale, high_qp, results_dir):

    classifier = Classifier()

    final_results = Results()

    dataset_root = Path(dds_env['dataset'])
    project_root = Path(dds_env['root'])
    lq_images_dir = dataset_root / f'{video}_{low_scale}_{low_qp}/src'
    assert os.path.exists(lq_images_dir)
    orig_images_dir = dataset_root / video / 'src'

    number_of_frames = len(glob.glob1(orig_images_dir, "*.png"))
    lq_images_path = [lq_images_dir / ('%010d.png' % i) for i in range(0, number_of_frames)]

    for idx, image_path in enumerate(lq_images_path):

        image = plt.imread(str(image_path))

        regions = classifier.region_proposal([image], idx, low_scale)

        for region in regions:
            final_results.append(region)


        classifier.logger.info(f'Region proposal for {image_path} completed.')

    os.system(f"mkdir -p {project_root / f'results_{video}'/ 'no_filter_combined_merged_bboxes'}")
    final_results.write(str(
        project_root /
        f'results_{video}' /
        'no_filter_combined_merged_bboxes'/
        f'{video}_mpeg_{low_scale}_{low_qp}'))

    os.system(f"mkdir -p {project_root / f'results_{video}'/ 'no_filter_combined_bboxes'}")
    final_results.write(str(
        project_root /
        f'results_{video}' /
        'no_filter_combined_bboxes'/
        f'{video}_mpeg_{low_scale}_{low_qp}'))

    print('RPN Done')

