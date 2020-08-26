import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

class EncoderCNN(nn.Module):
    def __init__(self, image_size=128,
                 latent_size=128,
                 filters=(128, 256, 256, 512),
                 conv=(5, 5, 5, 5),
                 stride=(2, 2, 2, 2)):
        super().__init__()
        out_channels = list(filters)
        in_channels = [3] + out_channels[:-1]
        self.convs = nn.ModuleList([
            nn.Conv2d(ic, oc, c, s, c // 2) for ic, oc, c, s in zip(in_channels, out_channels, conv, stride)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(oc) for oc in out_channels[:-1]])

        self.flatten = nn.Flatten()

        stride_factor = 1
        for s in stride:
            stride_factor *= s

        input_linear_size = int(filters[-1] * (image_size / stride_factor) ** 2)
        self.fc = nn.Linear(input_linear_size, latent_size)

    def forward(self, x):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x)
            # x = bn(x)
            x = F.relu(x)

        x = self.convs[-1](x)
        x = F.relu(x)
        x = self.flatten(x)
        return self.fc(x)

if __name__ == "__main__":
    enc = EncoderCNN(filters=(32, 32, 64, 64))
    enc.cuda()

    print(summary(enc, (3, 128, 128)))