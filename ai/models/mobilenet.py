import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class MobileNetBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        model = mobilenet_v2(weights="DEFAULT")

        # Remove classifier
        self.features = model.features

    def forward(self, x):

        x = self.features(x)

        return x