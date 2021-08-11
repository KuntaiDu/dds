import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as utils
import torchvision.transforms as T

import yaml
with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())

def swish(x):
    return x * torch.sigmoid(x)

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))

class SuperResoluter(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(SuperResoluter, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(self.upsample_factor//2):
            self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

        self.load_state_dict(torch.load(dds_env['super_resoluter']))

        self.eval().cuda(dds_env['super_resoluter_cuda_id'])

    def forward(self, x):
        x = swish(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(self.upsample_factor//2):
            x = self.__getattr__('upsample' + str(i+1))(x)

        return self.conv3(x)


class SRGAN():

    def __init__(self):
        self.sr = SuperResoluter(16, 4)

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

        self.unnormalize = T.Compose([
            T.Normalize(
                mean = [-2.118, -2.036, -1.804],
                std = [4.367, 4.464, 4.444])
        ])

    def __call__(self, image):

        # should be single RGB image
        assert len(image.shape) == 3

        image = self.normalize(image)
        image = image.cuda(dds_env['super_resoluter_cuda_id'])[None,:,:,:]
        image = self.sr(image).squeeze(0)
        image = self.unnormalize(image)

        # copy from torchvision.utils.save_image
        grid = utils.make_grid(
            image,
            nrow=8,
            padding=2,
            pad_value=0,
            normalize=False,
            range=None,
            scale_each = False)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1,2,0).to('cpu', torch.uint8).numpy()
        return ndarr

