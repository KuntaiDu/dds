
from . import carn

import torchvision.transforms.functional as F
import torch
from collections import OrderedDict

from PIL import Image

import yaml
with open('dds_env.yaml', 'r') as f:
    dds_env = yaml.load(f.read())

class CARN:

    def __init__(self, upscale = 4):
        self.net = carn.Net(multi_scale=True, group=1)
        self.upscale = upscale

        state_dict = torch.load(dds_env['super_resoluter'])
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            # name = k[7:] # remove "module."
            new_state_dict[name] = v

        self.net.load_state_dict(new_state_dict)
        self.net.cuda(dds_env['super_resoluter_cuda_id'] )

    def __call__(self, image):
        image = F.to_tensor(image)
        image = image[None, :, :, :].cuda(dds_env['super_resoluter_cuda_id'])
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            image = self.net(image, self.upscale).squeeze(0)
        image = image.mul(255).clamp(0, 255).byte().permute(1,2,0).cpu().numpy()
        image = Image.fromarray(image)
        # print(image)
        # input()
        return image