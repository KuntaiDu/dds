
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


    def region_proposal(self, image, fid, resolution, k = 63, topk = 3):

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
                results.append(Region(fid, (index[1] - k + 1) / y, (index[0] - k + 1) / x, k / y, k / x, 1.0, 'region proposal', resolution))
                set_zero(grad, index[0], index[1])

        assert k % 2 == 1

        image = self.transform(image)
        image = image.cuda()
        image.requires_grad = True
        loss = F.softmax(self.model(image[None,:,:,:]), dim=1).norm(2)
        loss.backward()
        grad = image.grad

        with torch.no_grad():
            grad = torch.abs(grad)
            grad = grad.sum(dim = 0).type(torch.DoubleTensor)
            grad = torch.cumsum(torch.cumsum(grad, axis = 0), axis = 1)
            grad_pad = F.pad(grad, (k,k,k,k))
            x, y = grad.shape
            grad_sum = grad[:, :] + grad_pad[0: x, 0:y] - grad_pad[k:x+k, 0:y] - grad_pad[0:x, k:y+k]

            results = []
            generate_regions(grad_sum, results)

        return results
