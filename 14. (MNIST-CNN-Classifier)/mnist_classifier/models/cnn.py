import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # |x| => (batch_size, in_channels, h, w)
        y = self.layers(x)
        # |y| => (batch_size, out_channels, h, w)

        return y


class ConvolutionClassifier(nn.Module):
    def __init__(self, output_size):

        self.output_size = output_size
        super().__init__()

        self.blocks = nn.Sequential(    # |x| => (n, 1, 28, 28)
            ConvolutionBlock(1, 32),    # |x| => (n, 32, 14, 14)
            ConvolutionBlock(32, 64),   # |x| => (n, 64, 7, 7)
            ConvolutionBlock(64, 128),  # |x| => (n, 128, 4, 4)
            ConvolutionBlock(128, 256),  # |x| => (n, 256, 2, 2)
            ConvolutionBlock(256, 512)  # |x| => (n, 512, 1, 1)
        )

        self.layers = nn.Sequential(
            nn.Linear(512, 256),  # |x| => (n, 256, 1, 1)
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),  # |x| => (n, 128, 1, 1)
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),  # |x| => (n, 64, 1, 1)
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        assert x.dim() > 2

        if x.dim() == 3:
            # |x| => (batch size, h, w)
            x = x.view(-1, 1, x.size(-2), x.size(-1))
        # |x| => (batch_size, 1, h, w)

        z = self.blocks(x)
        # |z| => (batch_size, 512, 1, 1)

        y = self.layers(z.squeeze())  # Input => (batch_size, 512)
        # |y| => (batch_size, output_size)

        return y
