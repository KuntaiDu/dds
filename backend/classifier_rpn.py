
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

import yaml
with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())

class RPN(nn.Module):

    def __init__(self, classifier):

        super(RPN, self).__init__()
        self.classifier = classifier
        self.upscale = [
            nn.PixelShuffle(1),
            nn.PixelShuffle(2),
            nn.PixelShuffle(4),
            nn.PixelShuffle(8),
            nn.PixelShuffle(16)
        ]
        self.convs = nn.Sequential(
            nn.BatchNorm2d(122),
            nn.Conv2d(122, 64, 3, padding=3//2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=3//2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=3//2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=3//2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=3//2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, 3, padding=3//2),
            nn.BatchNorm2d(1),
        )
        self.load_state_dict(torch.load(dds_env['classifier_rpn']))
        self.eval().cuda()

    def forward(self, features):

        x = [self.upscale[i](features[i]) for i in range(len(features))]
        x = torch.cat(x, dim=1)
        x = self.convs(x)
        x[x<0]=0
        x[x>1]=1
        return x