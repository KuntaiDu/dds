
import torch
from torchvision.models import vgg19_bn
import torchvision.transforms as T
import torch.nn.functional as F
from ..dds_utils import Region

class Classifier():

    def __init__(self):

        self.model = vgg19_bn(pretrained=True)
        self.model.eval().cuda()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def infer(self, image):

        with torch.no_grad():

            image = self.transform(image).cuda()
            result = F.softmax(self.model(image[None,:,:,:]), dim=1)[0]
            return result.cpu().data.numpy()


    def region_proposal(self, image, ksize = 31):

        def unravel_index(index, shape):
            out = []
            for dim in reversed(shape):
                out.append(index % dim)
                index = index // dim
            return tuple(reversed(out))

        def generate_regions(grad, k, results):
            for i in range(k):
                index = unravel_index(torch.argmax(grad), grad.shape)
                index = [int(index[0], index[1])]
                results.append(Region(


                ))
                grad[index[0]-15:index[0]+16, index[1]-15:index[1]+16] = 0

        assert ksize % 2 == 1

        with torch.enable_grad():

            image = self.transform(image)
            image.requires_grad = True
            image = image.cuda()

            loss = F.softmax(self.model(image[None,:,:,:])).norm(2)
            loss.backward()

        with torch.no_grad():

            grad = torch.abs(image.grad)
            grad = F.unfold(grad[None,:,:,:], ksize, padding = ksize//2)
            grad = torch.sum(grad, dim=1)
            grad = F.fold(grad[:, None, :], (720, 1280), 1)
            grad = grad[0,0,:,:]

            results = []


