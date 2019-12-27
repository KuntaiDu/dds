
import torch
import torch.nn as nn

import yaml
with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())

class RPN(nn.Module):

    def __init__(self, segmenter):

        super(RPN, self).__init__()
        self.segmenter = segmenter

        def upscale(r, C):
            return nn.Sequential(
                nn.PixelShuffle(r),
                nn.BatchNorm2d(C // (r*r))
            )

        self.upscale = nn.ModuleList([
            nn.PixelShuffle(4),
            nn.PixelShuffle(8),
            nn.PixelShuffle(8),
            nn.PixelShuffle(8),
            nn.PixelShuffle(1)
        ])
        self.convs = nn.Sequential(
            nn.BatchNorm2d(93),
            nn.Conv2d(93, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.BatchNorm2d(1)
        )
        self.load_state_dict(torch.load(dds_env['segmenter_rpn']))
        self.eval().cuda()

    def forward(self, features):
        x = torch.cat([self.upscale[i](features[i]) for i in range(len(features))], dim=1)
        x = self.convs(x)
        x[x<0] = 0
        x[x>1] = 1
        return x
