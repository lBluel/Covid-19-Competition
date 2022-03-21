import torchvision.models as models

from torch.nn import Module, Linear, Sequential, ReLU

class Alexnet(Module):
    def __init__(self, pretrained=False):
        super(Alexnet, self).__init__()
        self.model = models.alexnet(pretrained=pretrained)
        self.model.classifier[6] = Linear(4096, 500)
        self.model.classifier.append(ReLU())
        self.model.classifier.append(Linear(500,250))
        self.model.classifier.append(Linear(250,1))

    def forward(self, image):
        feature = self.model(image)
        return feature
