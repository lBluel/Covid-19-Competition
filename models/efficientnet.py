import torchvision.models as models

from torch.nn import Module, Linear, Sequential, ReLU, Dropout

class Efficientnet(Module):
    def __init__(self, pretrained=False):
        super(Efficientnet, self).__init__()
        self.model = models.efficientnet_b4(pretrained=pretrained)
        self.model.classifier = Sequential(
            Dropout(p=0.4, inplace=True),
            Linear(1792,500),
            ReLU(),
            Linear(500,250),
            Linear(250,1)
        )

    def forward(self, image):
        feature = self.model(image)
        return feature
