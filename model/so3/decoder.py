import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

class Decoder(nn.Module):
    def __init__(self, image_size=128,
                 latent_size=128,
                 filters=(128, 256, 256, 512),
                 conv=(5, 5, 5, 5),
                 stride=(2, 2, 2, 2)):
        super().__init__()
        in_channels = list(reversed(filters))
        out_channels = in_channels[1:] + [3]
        conv = list(reversed(conv))
        stride = list(reversed(stride))
        print(in_channels)
        print(out_channels)
        self.deconvs = nn.ModuleList([
            nn.Conv2d(in_channels=ic,
                      out_channels=oc,
                      kernel_size=c,
                      padding=c // 2) for ic, oc, c, s in zip(in_channels, out_channels, conv, stride)
        ])
        self.ups = nn.ModuleList([nn.Upsample(scale_factor=s, mode='nearest') for s in stride])
        self.bns = nn.ModuleList([nn.BatchNorm2d(oc) for oc in out_channels[:-1]])

        self.flatten = nn.Flatten()

        self.stride_factor = 1
        for s in stride:
            self.stride_factor *= s
        self.image_size = image_size
        self.last_filter = in_channels[0]

        output_linear_size = int(self.last_filter * (image_size / self.stride_factor) ** 2)
        self.fc = nn.Linear(latent_size, output_linear_size)

    def forward(self, x):
        x = self.fc(x)
        x = F.leaky_relu(x, 0.2)
        x = x.view((-1, self.last_filter, self.image_size // self.stride_factor, self.image_size // self.stride_factor))

        for conv, bn, up in zip(self.deconvs[:-1], self.bns, self.ups[:-1]):
            x = up(x)
            x = conv(x)
            x = F.leaky_relu(x, 0.2)
            x = bn(x)

        x = self.ups[-1](x)
        x = self.deconvs[-1](x)
        return torch.sigmoid(x)


if __name__ == "__main__":
    enc = Decoder(filters=(32, 32, 64, 64))
    enc.cuda()

    print(summary(enc, (1, 128)))