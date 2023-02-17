from __future__ import print_function
import torch
from torch.nn import Module, Conv2d, Dropout2d, Linear, Sequential, MaxPool2d, ReLU
import torch.nn.functional as F

from convolutional_network import ConvolutionalNetwork


class SpatialTransformer(ConvolutionalNetwork):
    def __init__(self, num_channels=2):
        super(SpatialTransformer, self).__init__(num_channels)

        # Spatial transformer localization-network
        self.localization = Sequential(
            Conv2d(num_channels, 8, kernel_size=7),
            MaxPool2d(2, stride=2),
            ReLU(True),
            Conv2d(8, 10, kernel_size=5),
            MaxPool2d(2, stride=2),
            ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_a = Sequential(
            Linear(10 * 4 * 12, 32),
            ReLU(True),
            Linear(32, 3 * 2)
        )

        self.fc_loc_b = Sequential(
            Linear(10 * 38 * 17, 32),
            ReLU(True),
            Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc_a[2].weight.data.zero_()
        self.fc_loc_a[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn_a(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 12)
        theta = self.fc_loc_a(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def stn_b(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 38 * 17)
        theta = self.fc_loc_b(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, xa, xb):
        # transform the input
        xa = self.stn_a(xa)
        xb = self.stn_b(xb)

        return super(SpatialTransformer, self).forward(xa, xb)