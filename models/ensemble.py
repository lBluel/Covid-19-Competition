import os
import torch
import torchvision.models as models

from torch.nn import Module, Linear, Sequential, ReLU

from models.alexnet import Alexnet
from models.densenet import Densenet
from models.efficientnet import Efficientnet
from models.resnext import Resnext


class Ensemble(Module):
    def __init__(self):
        super(Ensemble, self).__init__()
        self.alexnet = Alexnet()
        self.densenet = Densenet()
        self.efficientnet = Efficientnet()
        self.resnext = Resnext()
        self.model = Sequential(
            Linear(4, 1000),
            Linear(1000, 500),
            ReLU(),
            Linear(500, 250),
            ReLU(),
            Linear(250, 1)
        )

    def forward(self, image):
        alexnet_features = self.alexnet(image)
        densenet_features = self.densenet(image)
        efficientnet_features = self.efficientnet(image)
        resnext_features = self.resnext(image)
        feature = torch.stack([alexnet_features, densenet_features, efficientnet_features, resnext_features])
        feature = self.model(feature)
        return feature

    def load_ensemble_models(self, root_dir):
        checkpoint_alexnet = torch.load(os.path.join(root_dir, "alexnet", "checkpoint.tar"))
        self.alexnet.load_state_dict(checkpoint_alexnet['model_state_dict'])

        checkpoint_densenet = torch.load(os.path.join(root_dir, "densenet", "checkpoint.tar"))
        self.densenet.load_state_dict(checkpoint_densenet['model_state_dict'])

        checkpoint_efficientnet = torch.load(os.path.join(root_dir, "efficientnet", "checkpoint.tar"))
        self.efficientnet.load_state_dict(checkpoint_efficientnet['model_state_dict'])

        checkpoint_resnext = torch.load(os.path.join(root_dir, "resnext", "checkpoint.tar"))
        self.resnext.load_state_dict(checkpoint_resnext['model_state_dict'])
