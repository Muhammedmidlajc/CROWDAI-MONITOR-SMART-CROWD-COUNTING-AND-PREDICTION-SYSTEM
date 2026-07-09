import torch
import torch.nn as nn


class ConvBranch(nn.Module):

    def __init__(self, kernel_size):

        super().__init__()

        padding = kernel_size // 2

        self.branch = nn.Sequential(

            nn.Conv2d(1280, 256,
                      kernel_size=kernel_size,
                      padding=padding),

            nn.ReLU(inplace=True),

            nn.Conv2d(256,128,3,padding=1),

            nn.ReLU(inplace=True)

        )

    def forward(self,x):

        return self.branch(x)


class EnhancedMCCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.branch1 = ConvBranch(3)

        self.branch2 = ConvBranch(5)

        self.branch3 = ConvBranch(7)

    def forward(self,x):

        b1 = self.branch1(x)

        b2 = self.branch2(x)

        b3 = self.branch3(x)

        return b1,b2,b3