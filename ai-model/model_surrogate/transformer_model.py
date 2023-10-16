from transformer_layers import *


class SimpleTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=96,
                 pos_dim=1,
                 n_head=2,
                 dim_feedforward=512,
                 attention_type='fourier',
                 pos_emb=False,
                 layer_norm=True,
                 attn_norm=None,
                 norm_type='layer',
                 norm_eps=None,
                 batch_norm=False,
                 attn_weight=False,
                 xavier_init: float=1e-2,
                 diagonal_weight: float=1e-2,
                 symmetric_init=False,
                 residual_type='add',
                 activation_type='relu',
                 dropout=0.1,
                 ffn_dropout=None,
                 debug=False,
                 ):
        super(SimpleTransformerEncoderLayer, self).__init__()

        dropout = default(dropout, 0.05)
        if attention_type in ['linear', 'softmax']:
            dropout = 0.1
        ffn_dropout = default(ffn_dropout, dropout)
        norm_eps = default(norm_eps, 1e-5)
        attn_norm = default(attn_norm, not layer_norm)
        if (not layer_norm) and (not attn_norm):
            attn_norm = True
        norm_type = default(norm_type, 'layer')

        self.attn = SimpleAttention(n_head=n_head,
                                    d_model=d_model,
                                    attention_type=attention_type,
                                    diagonal_weight=diagonal_weight,
                                    xavier_init=xavier_init,
                                    symmetric_init=symmetric_init,
                                    pos_dim=pos_dim,
                                    norm=attn_norm,
                                    norm_type=norm_type,
                                    eps=norm_eps,
                                    dropout=dropout)
        self.d_model = d_model
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.add_layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        dim_feedforward = default(dim_feedforward, 2*d_model)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              batch_norm=batch_norm,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_type = residual_type  # plus or minus
        self.add_pos_emb = pos_emb
        if self.add_pos_emb:
            self.pos_emb = PositionalEncoding(d_model)

        self.debug = debug
        self.attn_weight = attn_weight
        self.__name__ = attention_type.capitalize() + 'TransformerEncoderLayer'

    def forward(self, x, pos=None, weight=None):
        '''
        - x: node feature, (batch_size, seq_len, n_feats)
        - pos: position coords, needed in every head
        Remark:
            - for n_head=1, no need to encode positional
            information if coords are in features
        '''
        if self.add_pos_emb:
            x = x.permute((1, 0, 2))
            x = self.pos_emb(x)
            x = x.permute((1, 0, 2))

        if pos is not None and self.pos_dim > 0:
            att_output, attn_weight = self.attn(
                x, x, x, pos=pos, weight=weight)  # encoder no mask
        else:
            att_output, attn_weight = self.attn(x, x, x, weight=weight)

        x = x.view(x.size(0), x.size(1), -1)

        if self.residual_type in ['add', 'plus'] or self.residual_type is None:
            x = x + self.dropout1(att_output)
        else:
            x = x - self.dropout1(att_output)
        if self.add_layer_norm:
            x = self.layer_norm1(x)

        x1 = self.ff(x)
        x = x + self.dropout2(x1)

        if self.add_layer_norm:
            x = self.layer_norm2(x)

        if self.attn_weight:
            return x, attn_weight
        else:
            return x


class GCN(nn.Module):
    def __init__(self,
                 node_feats=4,
                 out_features=96,
                 num_gcn_layers=2,
                 edge_feats=6,
                 activation=True,
                 raw_laplacian=False,
                 dropout=0.1,
                 debug=False):
        super(GCN, self).__init__()
        '''
        A simple GCN, a wrapper for Kipf and Weiling's code
        learnable edge features similar to 
        Graph Transformer https://arxiv.org/abs/1911.06455
        but using neighbor agg
        '''
        self.edge_learner = EdgeEncoder(out_dim=out_features,
                                        edge_feats=edge_feats,
                                        raw_laplacian=raw_laplacian
                                        )
        self.gcn_layer0 = GraphConvolution(in_features=node_feats,  # hard coded
                                           out_features=out_features,
                                           debug=debug,
                                           )
        self.gcn_layers = nn.ModuleList([copy.deepcopy(GraphConvolution(
            in_features=out_features,  # hard coded
            out_features=out_features,
            debug=debug
        )) for _ in range(1, num_gcn_layers)])
        self.activation = activation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.edge_feats = edge_feats
        self.debug = debug

    def forward(self, x, edge):
        x = x.permute(0, 2, 1).contiguous()
        edge = edge.permute([0, 3, 1, 2]).contiguous()
        assert edge.size(1) == self.edge_feats

        edge = self.edge_learner(edge)

        out = self.gcn_layer0(x, edge)
        for gc in self.gcn_layers[:-1]:
            out = gc(out, edge)
            if self.activation:
                out = self.relu(out)

        # last layer no activation
        out = self.gcn_layers[-1](out, edge)
        return out.permute(0, 2, 1)


class GAT(nn.Module):
    def __init__(self,
                 node_feats=4,
                 out_features=96,
                 num_gcn_layers=2,
                 edge_feats=None,
                 activation=False,
                 debug=False):
        super(GAT, self).__init__()
        '''
        A simple GAT: modified from the official implementation
        '''
        self.gat_layer0 = GraphAttention(in_features=node_feats,
                                         out_features=out_features,
                                         )
        self.gat_layers = nn.ModuleList([copy.deepcopy(GraphAttention(
            in_features=out_features,
            out_features=out_features,
        )) for _ in range(1, num_gcn_layers)])
        self.activation = activation
        self.relu = nn.ReLU()
        self.debug = debug

    def forward(self, x, edge):
        '''
        input: node feats (-1, seq_len, n_feats)
               edge only takes adj (-1, seq_len, seq_len)
               edge matrix first one in the last dim is graph Lap.
        '''
        edge = edge[..., 0].contiguous()

        out = self.gat_layer0(x, edge)

        for layer in self.gat_layers[:-1]:
            out = layer(out, edge)
            if self.activation:
                out = self.relu(out)

        # last layer no activation
        return self.gat_layers[-1](out, edge)


class PointwiseRegressor(nn.Module):
    def __init__(self, in_dim,  # input dimension
                 n_hidden,
                 out_dim,  # number of target dim
                 num_layers: int = 2,
                 spacial_fc: bool = False,
                 spacial_dim=1,
                 dropout=0.1,
                 activation='silu',
                 return_latent=False,
                 debug=False):
        super(PointwiseRegressor, self).__init__()
        '''
        A wrapper for a simple pointwise linear layers
        '''
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc
        activ = nn.SiLU() if activation == 'silu' else nn.ReLU()
        if self.spacial_fc:
            in_dim = in_dim + spacial_dim
            self.fc = nn.Linear(in_dim, n_hidden)
        self.ff = nn.ModuleList([nn.Sequential(
                                nn.Linear(n_hidden, n_hidden),
                                activ,
                                )])
        for _ in range(num_layers - 1):
            self.ff.append(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                activ,
            ))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_hidden, out_dim)
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)

        for layer in self.ff:
            x = layer(x)
            x = self.dropout(x)

        x = self.out(x)

        if self.return_latent:
            return x, None
        else:
            return x


class SpectralRegressor(nn.Module):
    def __init__(self, in_dim,
                 n_hidden,
                 freq_dim,
                 out_dim,
                 modes: int,
                 num_spectral_layers: int = 2,
                 n_grid=None,
                 dim_feedforward=None,
                 spacial_fc=False,
                 spacial_dim=2,
                 return_freq=False,
                 return_latent=False,
                 normalizer=None,
                 activation='silu',
                 last_activation=True,
                 dropout=0.1,
                 debug=False):
        super(SpectralRegressor, self).__init__()
        '''
        A wrapper for both SpectralConv1d and SpectralConv2d
        Ref: Li et 2020 FNO paper
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        A new implementation incoporating all spacial-based FNO
        in_dim: input dimension, (either n_hidden or spacial dim)
        n_hidden: number of hidden features out from attention to the fourier conv
        '''
        if spacial_dim == 2:  # 2d, function + (x,y)
            spectral_conv = SpectralConv2d
        #elif spacial_dim == 1:  # 1d, function + x
        #    spectral_conv = SpectralConv1d
        else:
            raise NotImplementedError("3D not implemented.")
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc  # False in Transformer
        if self.spacial_fc:
            self.fc = nn.Linear(in_dim + spacial_dim, n_hidden)
        self.spectral_conv = nn.ModuleList([spectral_conv(in_dim=n_hidden,
                                                          out_dim=freq_dim,
                                                          n_grid=n_grid,
                                                          modes=modes,
                                                          dropout=dropout,
                                                          activation=activation,
                                                          return_freq=return_freq,
                                                          debug=debug)])
        for _ in range(num_spectral_layers - 1):
            self.spectral_conv.append(spectral_conv(in_dim=freq_dim,
                                                    out_dim=freq_dim,
                                                    n_grid=n_grid,
                                                    modes=modes,
                                                    dropout=dropout,
                                                    activation=activation,
                                                    return_freq=return_freq,
                                                    debug=debug))
        if not last_activation:
            self.spectral_conv[-1].activation = Identity()

        self.n_grid = n_grid  # dummy for debug
        self.dim_feedforward = default(dim_feedforward, 2*spacial_dim*freq_dim)
        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, self.dim_feedforward),
            self.activation,
            nn.Linear(self.dim_feedforward, out_dim),
        )
        self.normalizer = normalizer
        self.return_freq = return_freq
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, edge=None, pos=None, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        x_latent = []
        x_fts = []

        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)

        for layer in self.spectral_conv:
            if self.return_freq:
                x, x_ft = layer(x)
                x_fts.append(x_ft.contiguous())
            else:
                x = layer(x)

            if self.return_latent:
                x_latent.append(x.contiguous())

        x = self.regressor(x)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)

        if self.return_freq or self.return_latent:
            return x, dict(preds_freq=x_fts, preds_latent=x_latent)
        else:
            return x


class DownScaler(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 dropout=0.1,
                 padding=5,
                 downsample_mode='conv',
                 activation_type='silu',
                 interp_size=None,
                 debug=False):
        super(DownScaler, self).__init__()
        '''
        A wrapper for conv2d/interp downscaler
        '''
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


class UpScaler(nn.Module):
    def __init__(self, in_dim: int,
                 out_dim: int,
                 hidden_dim=None,
                 padding=2,
                 output_padding=0,
                 dropout=0.1,
                 upsample_mode='conv',
                 activation_type='silu',
                 interp_mode='bilinear',
                 interp_size=None,
                 debug=False):
        super(UpScaler, self).__init__()
        '''
        A wrapper for DeConv2d upscaler or interpolation upscaler
        Deconv: Conv1dTranspose
        Interp: interp->conv->interp
        '''
        hidden_dim = default(hidden_dim, in_dim)
        if upsample_mode in ['conv', 'deconv']:
            self.upsample = nn.Sequential(
                DeConv2dBlock(in_dim=in_dim,
                              out_dim=out_dim,
                              hidden_dim=hidden_dim,
                              padding=padding,
                              output_padding=output_padding,
                              dropout=dropout,
                              activation_type=activation_type,
                              debug=debug),
                DeConv2dBlock(in_dim=in_dim,
                              out_dim=out_dim,
                              hidden_dim=hidden_dim,
                              padding=padding*2,
                              output_padding=output_padding,
                              dropout=dropout,
                              activation_type=activation_type,
                              debug=debug))
        elif upsample_mode == 'interp':
            self.upsample = Interp2dUpsample(in_dim=in_dim,
                                             out_dim=out_dim,
                                             interp_mode=interp_mode,
                                             interp_size=interp_size,
                                             dropout=dropout,
                                             activation_type=activation_type,
                                             debug=debug)
        else:
            raise NotImplementedError("upsample mode not implemented.")
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        '''
        2D:
            Input: (-1, n_s, n_s, in_dim)
            Output: (-1, n, n, out_dim)
        '''
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)
        return x
