import torch
import torch.nn as nn
import torch.nn.functional as F


def default(value, d):
    """Helper taken from https://github.com/lucidrains/linear-attention-transformer"""
    return d if value is None else value


class Shortcut2d(nn.Module):
    '''
    (-1, in, S, S) -> (-1, out, S, S)
    Used in SimpleResBlock
    '''

    def __init__(self, in_features=None,
                 out_features=None,):
        super(Shortcut2d, self).__init__()
        self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x, edge=None, grid=None):
        x = x.permute(0, 2, 3, 1)
        x = self.shortcut(x)
        x = x.permute(0, 3, 1, 2)
        return x


class Identity(nn.Module):
    '''
    a placeholder layer similar to tensorflow.no_op():
    https://github.com/pytorch/pytorch/issues/9160#issuecomment-483048684
    not used anymore as
    https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
    edge and grid are dummy inputs
    '''

    def __init__(self, in_features=None, out_features=None,
                 *args, **kwargs):
        super(Identity, self).__init__()

        if in_features is not None and out_features is not None:
            self.id = nn.Linear(in_features, out_features)
        else:
            self.id = nn.Identity()

    def forward(self, x, edge=None, grid=None):
        return self.id(x)


class Conv2dResBlock(nn.Module):
    '''Conv2d + a residual block
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    Modified from ResNet's basic block, one conv less, no batchnorm
    No batchnorm
    '''

    def __init__(self, in_dim, out_dim,
                 kernel_size=3,
                 padding=1,
                 dilation=1,
                 dropout=0.1,
                 stride=1,
                 bias=False,
                 residual=False,
                 basic_block=False,
                 activation_type='silu'):
        super(Conv2dResBlock, self).__init__()

        activation_type = default(activation_type, 'silu')
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.add_res = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim,
                      kernel_size=kernel_size,
                      padding=padding,
                      dilation=dilation,
                      stride=stride,
                      bias=bias),
            nn.Dropout(dropout),
        )
        self.basic_block = basic_block
        if self.basic_block:
            self.conv1 = nn.Sequential(
                self.activation,
                nn.Conv2d(out_dim, out_dim,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=bias),
                nn.Dropout(dropout),
            )
        self.apply_shortcut = (in_dim != out_dim)

        if self.add_res:
            if self.apply_shortcut:
                self.res = Shortcut2d(in_dim, out_dim)
            else:
                self.res = Identity()

    def forward(self, x):
        if self.add_res:
            h = self.res(x)

        x = self.conv(x)

        if self.basic_block:
            x = self.conv1(x)

        if self.add_res:
            return self.activation(x + h)
        else:
            return self.activation(x)


class Conv2dEncoder(nn.Module):
    """old code: first conv then pool.
    Similar to a LeNet block. \approx 1/4 subsampling"""

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 scaling_factor: int = 2,
                 residual=False,
                 activation_type='silu',
                 debug=False):
        super(Conv2dEncoder, self).__init__()

        conv_dim0 = out_dim // 3
        conv_dim1 = out_dim // 3
        conv_dim2 = int(out_dim - conv_dim0 - conv_dim1)
        padding1 = padding // 2 if padding // 2 >= 1 else 1
        padding2 = padding // 4 if padding // 4 >= 1 else 1
        activation_type = default(activation_type, 'silu')
        self.conv0 = Conv2dResBlock(in_dim, out_dim, kernel_size=kernel_size,
                                    padding=padding,
                                    residual=residual)
        self.conv1 = Conv2dResBlock(out_dim, conv_dim0, kernel_size=kernel_size,
                                    padding=padding1,
                                    stride=stride, residual=residual)
        self.conv2 = Conv2dResBlock(conv_dim0, conv_dim1, kernel_size=kernel_size,
                                    dilation=dilation,
                                    padding=padding2, residual=residual)
        self.conv3 = Conv2dResBlock(conv_dim1, conv_dim2,
                                    kernel_size=kernel_size,
                                    residual=residual)
        self.pool0 = nn.AvgPool2d(kernel_size=scaling_factor,
                                  stride=scaling_factor)
        self.pool1 = nn.AvgPool2d(kernel_size=scaling_factor,
                                  stride=scaling_factor)
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        # self.activation = nn.LeakyReLU() # leakyrelu decreased performance 10 times?
        self.debug = debug

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool0(x)
        x = self.activation(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        out = torch.cat([x1, x2, x3], dim=1)
        out = self.pool1(out)
        out = self.activation(out)
        return out


class Interp2dEncoder(nn.Module):
    r'''
    Using Interpolate instead of avg pool
    interp dim hard coded or using a factor
    old code uses lambda and cannot be pickled
    '''

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 interp_size=None,
                 residual=False,
                 activation_type='silu',
                 dropout=0.1,
                 debug=False):
        super(Interp2dEncoder, self).__init__()

        conv_dim0 = out_dim // 3
        conv_dim1 = out_dim // 3
        conv_dim2 = int(out_dim - conv_dim0 - conv_dim1)
        padding1 = padding // 2 if padding // 2 >= 1 else 1
        padding2 = padding // 4 if padding // 4 >= 1 else 1
        activation_type = default(activation_type, 'silu')
        self.interp_size = interp_size
        self.is_scale_factor = isinstance(
            interp_size[0], float) and isinstance(interp_size[1], float)
        self.conv0 = Conv2dResBlock(in_dim, out_dim, kernel_size=kernel_size,
                                    padding=padding, activation_type=activation_type,
                                    dropout=dropout,
                                    residual=residual)
        self.conv1 = Conv2dResBlock(out_dim, conv_dim0, kernel_size=kernel_size,
                                    padding=padding1,
                                    stride=stride, residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.conv2 = Conv2dResBlock(conv_dim0, conv_dim1, kernel_size=kernel_size,
                                    dilation=dilation,
                                    padding=padding2, residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.conv3 = Conv2dResBlock(conv_dim1, conv_dim2,
                                    kernel_size=kernel_size,
                                    residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.add_res = residual
        self.debug = debug

    def forward(self, x):
        x = self.conv0(x)
        if self.is_scale_factor:
            x = F.interpolate(x, scale_factor=self.interp_size[0],
                              mode='bilinear',
                              recompute_scale_factor=True,
                              align_corners=True)
        else:
            x = F.interpolate(x, size=self.interp_size[0],
                              mode='bilinear',
                              align_corners=True)
        x = self.activation(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat([x1, x2, x3], dim=1)
        if self.add_res:
            out += x

        if self.is_scale_factor:
            out = F.interpolate(out, scale_factor=self.interp_size[1],
                                mode='bilinear',
                                recompute_scale_factor=True,
                                align_corners=True,)
        else:
            out = F.interpolate(out, size=self.interp_size[1],
                              mode='bilinear',
                              align_corners=True)
        out = self.activation(out)
        return out


class DownScaler(nn.Module):
    """A wrapper for conv2d/interp downscaler"""

    def __init__(self, in_dim,
                 out_dim,
                 dropout=0.1,
                 padding=5,
                 downsample_mode='conv',
                 activation_type='silu',
                 interp_size=None,
                 debug=False):
        super(DownScaler, self).__init__()

        if downsample_mode == 'conv':
            self.downsample = nn.Sequential(Conv2dEncoder(in_dim=in_dim,
                                                          out_dim=out_dim,
                                                          activation_type=activation_type,
                                                          debug=debug),
                                            Conv2dEncoder(in_dim=out_dim,
                                                          out_dim=out_dim,
                                                          padding=padding,
                                                          activation_type=activation_type,
                                                          debug=debug))
        elif downsample_mode == 'interp':
            self.downsample = Interp2dEncoder(in_dim=in_dim,
                                              out_dim=out_dim,
                                              interp_size=interp_size,
                                              activation_type=activation_type,
                                              dropout=dropout,
                                              debug=debug)
        else:
            raise NotImplementedError("downsample mode not implemented.")
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        '''
        2D:
            Input: (-1, n, n, in_dim)
            Output: (-1, n_s, n_s, out_dim)
        '''
        n_grid = x.size(1)
        bsz = x.size(0)
        x = x.view(bsz, n_grid, n_grid, self.in_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        x = x.permute(0, 2, 3, 1)
        return x