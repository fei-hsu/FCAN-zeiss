import torch.nn.functional as F
import torch.nn as nn
import torch

from .aspp_layer import *

class RAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RAN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.aspp1 = ASPP(in_ch=self.in_channels, out_ch=self.out_channels, rates=[1])
        self.aspp2 = ASPP(in_ch=self.in_channels, out_ch=self.out_channels, rates=[2])
        self.aspp3 = ASPP(in_ch=self.in_channels, out_ch=self.out_channels, rates=[3])
        self.aspp4 = ASPP(in_ch=self.in_channels, out_ch=self.out_channels, rates=[4])

        self.convsig = nn.Sequential(
            nn.Conv2d(in_channels=4*self.out_channels, out_channels=1, kernel_size=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        feature_map = torch.cat((x1, x2, x3, x4), dim=1)
        output = self.convsig(feature_map)
        return output
