import torchvision.models as models

from torch.nn import Module, Linear, Sequential, ReLU

class Densenet(Module):
    def __init__(self):
        super(Densenet, self).__init__()
        self.model = models.densenet161()
        self.model.classifier = Sequential(
            Linear(2208,500),
            ReLU(),
            Linear(500,250),
            Linear(250,1)
        )

    def forward(self, image):
        feature = self.model(image)
        return feature
