import torch
import torch.nn as nn
import torch.nn.functional as F

from .mobilenet import MobileNetBackbone
from .mccnn import EnhancedMCCNN
from .fusion import FeatureFusion


class CrowdCounter(nn.Module):

    def __init__(self):
        super().__init__()

        # MobileNet Backbone
        self.backbone = MobileNetBackbone()

        # Enhanced MC-CNN
        self.mccnn = EnhancedMCCNN()

        # Feature Fusion
        self.fusion = FeatureFusion()

        # Density Map Prediction
        self.output_layer = nn.Sequential(

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1)

        )

    def forward(self, x):

        # Save original image size
        original_size = x.shape[2:]

        # MobileNet
        features = self.backbone(x)

        # Enhanced MC-CNN
        b1, b2, b3 = self.mccnn(features)

        # Feature Fusion
        fused = self.fusion(b1, b2, b3)

        # Density Prediction
        density = self.output_layer(fused)

        # Resize to original image size
        density = F.interpolate(
            density,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )

        return density