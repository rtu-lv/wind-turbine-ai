# import the necessary packages
from torch import cat
from torch import flatten
from torch.nn import Module, Conv2d, Linear, MaxPool2d, LeakyReLU, AdaptiveMaxPool2d, BatchNorm1d


def _init_weights(module):
    if isinstance(module, Linear):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, Conv2d):
        module.weight.data.normal_(mean=0.5, std=0.1)
        if module.bias is not None:
            module.bias.data.zero_()


class ConvolutionalNetwork_HR(Module):

    def __init__(self, num_channels):
        # call the parent constructor
        super(ConvolutionalNetwork_HR, self).__init__()

        self.activConv = LeakyReLU()
        self.max_pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.rect_pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.activFC = LeakyReLU()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=num_channels, out_channels=20,
                            kernel_size=(5, 5), stride=(2, 2), padding=(4, 4))

        # initialize second set of CONV (=> RELU => POOL layers)
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
                             kernel_size=(5, 5), stride=(2, 2), padding=(4, 4))

        # initialize separate set of FC => RELU layers
        self.fc = Linear(in_features=3000, out_features=200)


        self.bn = BatchNorm1d(num_features=200)

        # get output layer as a single value
        self.fcOut = Linear(in_features=200, out_features=1)

        # self.apply(self._init_weights)

    def forward(self, xa):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        # IN:  xa == (25 x 49)x2    xb == (169 x 49)x2
        # OUT: xa == (11 x 23)x20   xb == (27 x 23)x20
        xa = self.conv1(xa)  # 5x5 kernel, 2x2 stride
        xa = self.activConv(xa)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        # IN:  xa == (11 x 23)x20   xb == (27 x 23)x20
        # OUT: xa == (4 x 5)x50     xb == (6 x 5)x50
        xa = self.conv2(xa)  # 4x5 kernel, 1x2 stride
        xa = self.activConv(xa)
        xa = self.max_pool(xa)  # 2x2 pooling

        # flatten the output from the previous layer
        # IN:  xa == (4 x 5)x50     xb == (6 x 5)x50
        # OUT: xa == 1000           xb == 1500
        # print(xa.size(), xb.size())
        xa = flatten(xa, 1)  # flatten on dim 1 to account for batch dim 0


        # Concatenate the 2 tensors into one and pass them through
        # 2 set of set of FC => RELU layers
        # IN:  xa == 200        xb == 300
        # OUT: x == 100
        x = self.fc(xa)
        x = self.bn(x)
        x = self.activFC(x)

        # pass the output to final layer to get our output predictions
        # IN:  x == 100
        # OUT: x == 1
        x = self.fcOut(x)
        output = self.activFC(x)

        # return the output regression
        return output
