import torch
import torch.nn as nn


class FeatureFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.fusion = nn.Sequential(

            nn.Conv2d(384, 128, kernel_size=1),

            nn.ReLU(inplace=True)

        )

    def forward(self, b1, b2, b3):

        x = torch.cat([b1, b2, b3], dim=1)

        x = self.fusion(x)

        return x