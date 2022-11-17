# import the necessary packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear, Sigmoid, Tanh
from torch.nn import MaxPool2d
from torch.nn import ReLU, LeakyReLU
from torch import flatten
from torch import cat


class aLNetB(Module):

    def __init__(self, numChannels):
        # call the parent constructor
        super(aLNetB, self).__init__()

        self.activConv = LeakyReLU()
        self.maxpool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.activFC = LeakyReLU()
        self.activOut = Sigmoid()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
                            kernel_size=(5, 5), stride=(2, 2))

        # initialize second set of CONV (=> RELU => POOL layers)
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
                            kernel_size=(4, 5), stride=(1, 2))

        # initialize separate set of FC => RELU layers
        self.fc = Linear(in_features=1001, out_features=200)

        # get output layer as a single value
        self.fcOut = Linear(in_features=200, out_features=1)

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Conv2d):
            module.weight.data.normal_(mean=0.5, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, ang):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        # IN:  x == (25 x 49)x2
        # OUT: x == (11 x 23)x20
        x = self.conv1(x)  # 5x5 kernel, 2x2 stride
        x = self.activConv(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        # IN:  x == (11 x 23)x20
        # OUT: x == (4 x 5)x50
        x = self.conv2(x)  # 4x5 kernel, 1x2 stride
        x = self.activConv(x)
        x = self.maxpool(x)  # 2x2 pooling

        # flatten the output from the previous layer
        # IN:  x == (4 x 5)x50
        # OUT: x == 1000
        x = flatten(x, 1)  # flatten on dim 1 to account for batch dim 0

        # Add angle value to the flattened arrays
        # IN:  x == 1000
        # OUT: x == 1001
        x = cat((x, ang), 1)  # Concatenate on dim 1 to account for batch dim 0

        # pass the output through 2 separate sets of FC => RELU layers
        # IN:  x == 1001
        # OUT: x == 200
        x = self.fc(x)
        x = self.activFC(x)

        # pass the output to final layer to get our output predictions
        # IN:  x == 200
        # OUT: x == 1
        x = self.fcOut(x)
        output = self.activOut(x)

        # return the output regression
        return output