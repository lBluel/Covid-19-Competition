import torchvision.models as models

from torch.nn import Module, Linear, Sequential, ReLU

class Resnext(Module):
    def __init__(self, pretrained=False):
        super(Resnext, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=pretrained)
        self.model.classifier = Sequential(
            Linear(2048,500),
            ReLU(),
            Linear(500,250),
            Linear(250,1)
        )

    def forward(self, image):
        feature = self.model(image)
        return feature
