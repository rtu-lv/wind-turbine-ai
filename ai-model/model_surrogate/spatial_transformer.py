from __future__ import print_function
import torch
from torch.nn import Module, Conv2d, Dropout2d, Linear, Sequential, MaxPool2d, ReLU
import torch.nn.functional as F


class SpatialTransformer(Module):
    def __init__(self, num_channels=2):
        super(SpatialTransformer, self).__init__()
        self.conv1 = Conv2d(1, 10, kernel_size=5)
        self.conv2 = Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = Dropout2d()
        self.fc1 = Linear(320, 50)
        self.fc2 = Linear(50, 10)

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
        self.fc_loc = Sequential(
            Linear(15360, 32),
            ReLU(True),
            Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 15360)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, xa, xb):
        # transform the input
        x = self.stn(xa)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)