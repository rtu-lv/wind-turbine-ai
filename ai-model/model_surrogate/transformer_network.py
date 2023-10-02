# import the necessary packages
from torch import cat
from torch import flatten
from torch.nn import Module, ModuleList, Conv2d, Linear, MaxPool2d, LeakyReLU, BatchNorm1d
from torch.nn import TransformerEncoderLayer
from transformer_model import SimpleTransformerEncoderLayer
import copy

def _init_weights(module):
    if isinstance(module, Linear):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, Conv2d):
        module.weight.data.normal_(mean=0.5, std=0.1)
        if module.bias is not None:
            module.bias.data.zero_()


class TransformerNetwork(Module):

    def __init__(self, config, num_channels):
        # call the parent constructor
        super(TransformerNetwork, self).__init__()

        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        self._get_encoder()

        self.activConv = LeakyReLU()
        self.max_pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.rect_pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.activFC = LeakyReLU()

        conv1a_kernel_size = 5
        conv1b_kernel_size = 5

        # initialize first set of CONV => RELU => POOL layers
        self.conv1a = Conv2d(in_channels=num_channels, out_channels=20,
                             kernel_size=(conv1a_kernel_size, conv1a_kernel_size), stride=(2, 2),
                             padding=(conv1a_kernel_size - 1, conv1a_kernel_size - 1))
        self.conv1b = Conv2d(in_channels=num_channels, out_channels=20,
                             kernel_size=(conv1b_kernel_size, conv1b_kernel_size), stride=(2, 2),
                             padding=(conv1b_kernel_size - 1, conv1b_kernel_size - 1))

        conv2a_out_channels = config["conv2a_out_channels"]
        conv2b_out_channels = config["conv2b_out_channels"]

        conv2a_kernel_size = 5
        conv2b_kernel_size = 5

        # initialize second set of CONV (=> RELU => POOL layers)
        self.conv2a = Conv2d(in_channels=20, out_channels=conv2a_out_channels,
                             kernel_size=(conv2a_kernel_size, conv2a_kernel_size), stride=(2, 2),
                             padding=(conv2a_kernel_size - 1, conv2a_kernel_size - 1))
        self.conv2b = Conv2d(in_channels=20, out_channels=conv2b_out_channels,
                             kernel_size=(conv2b_kernel_size, conv2b_kernel_size), stride=(2, 2),
                             padding=(conv2b_kernel_size - 1, conv2b_kernel_size - 1))

        fca_out_features = config["fca_out_features"]
        fcb_out_features = config["fcb_out_features"]

        # initialize separate set of FC => RELU layers
        self.fca = Linear(in_features=30*conv2a_out_channels*num_channels, out_features=fca_out_features)
        self.fcb = Linear(in_features=36*conv2b_out_channels*num_channels, out_features=fcb_out_features)

        fc1_out_features = config["fc1_out_features"]

        # initialize first common linear layer FC => RELU
        self.fc1 = Linear(in_features=fca_out_features+fcb_out_features, out_features=fc1_out_features)

        self.bn = BatchNorm1d(num_features=fc1_out_features)

        # get output layer as a single value
        self.fcOut = Linear(in_features=fc1_out_features, out_features=4)

        # self.apply(self._init_weights)

    def forward(self, xa, xb):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        # IN:  xa == (25 x 49)x2    xb == (169 x 49)x2
        # OUT: xa == (11 x 23)x20   xb == (27 x 23)x20
        xa = self.conv1a(xa)  # 5x5 kernel, 2x2 stride
        xa = self.activConv(xa)
        xb = self.conv1b(xb)  # 9x5 kernel, 2x2 stride
        xb = self.activConv(xb)
        xb = self.rect_pool(xb)  # rectangular 3x1 pooling

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        # IN:  xa == (11 x 23)x20   xb == (27 x 23)x20
        # OUT: xa == (4 x 5)x50     xb == (6 x 5)x50
        xa = self.conv2a(xa)  # 4x5 kernel, 1x2 stride
        xa = self.activConv(xa)
        xa = self.max_pool(xa)  # 2x2 pooling
        xb = self.conv2b(xb)  # 5x5 kernel, 2x2 stride
        xb = self.activConv(xb)
        xb = self.max_pool(xb)  # 2x2 pooling

        # flatten the output from the previous layer
        # IN:  xa == (4 x 5)x50     xb == (6 x 5)x50
        # OUT: xa == 1000           xb == 1500
        # print(xa.size(), xb.size())
        xa = flatten(xa, 1)  # flatten on dim 1 to account for batch dim 0
        xb = flatten(xb, 1)

        # pass the output through 2 separate sets of FC => RELU layers
        # IN:  xa == 1001           xb == 1501
        # OUT: xa == 200            xb == 300
        xa = self.fca(xa)
        xa = self.activFC(xa)
        xb = self.fcb(xb)
        xb = self.activFC(xb)

        # Concatenate the 2 tensors into one and pass them through
        # 2 set of set of FC => RELU layers
        # IN:  xa == 200        xb == 300
        # OUT: x == 100
        x = cat((xa, xb), 1)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.activFC(x)

        # pass the output to final layer to get our output predictions
        # IN:  x == 100
        # OUT: x == 1
        output = self.fcOut(x)

        # return the output regression
        return output

    def _get_encoder(self):
        if self.attention_type in self.attention_types:
            encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                           n_head=self.n_head,
                                                           attention_type=self.attention_type,
                                                           dim_feedforward=self.dim_feedforward,
                                                           layer_norm=self.layer_norm,
                                                           attn_norm=self.attn_norm,
                                                           batch_norm=self.batch_norm,
                                                           pos_dim=self.pos_dim,
                                                           xavier_init=self.xavier_init,
                                                           diagonal_weight=self.diagonal_weight,
                                                           symmetric_init=self.symmetric_init,
                                                           attn_weight=self.return_attn_weight,
                                                           dropout=self.encoder_dropout,
                                                           ffn_dropout=self.ffn_dropout,
                                                           norm_eps=self.norm_eps,
                                                           debug=self.debug)
        elif self.attention_type == 'official':
            encoder_layer = TransformerEncoderLayer(d_model=self.n_hidden+self.pos_dim*self.n_head,
                                                    nhead=self.n_head,
                                                    dim_feedforward=self.dim_feedforward,
                                                    dropout=self.encoder_dropout,
                                                    batch_first=True,
                                                    layer_norm_eps=self.norm_eps,
                                                    )
        else:
            raise NotImplementedError("encoder type not implemented.")
        self.encoder_layers = ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])