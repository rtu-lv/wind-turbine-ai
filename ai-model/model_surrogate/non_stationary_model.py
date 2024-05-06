import torch
import torch.nn as nn
from convolutional_network import ConvolutionalNetwork


class NonStationaryModel(nn.Module):
    def __init__(self, config, num_channels=2, time_steps=20, recurrent_type='RNN'):
        super(NonStationaryModel, self).__init__()

        print("Initializing nonstationary model")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.time_steps = time_steps
        self.cnn_models = nn.ModuleList(
            [ConvolutionalNetwork(config, num_channels=num_channels) for i in range(self.time_steps)])

        self.num_layers_rnn = 1
        self.hidden_size_rnn = 2
        self.input_size_rnn = config["cnn_out_features"]

        match recurrent_type:
            case "RNN":
                self.rnn = nn.RNN(input_size=self.input_size_rnn, hidden_size=self.hidden_size_rnn,
                                  num_layers=self.num_layers_rnn, batch_first=True)
            case "LSTM":
                self.rnn = nn.LSTM(input_size=self.input_size_rnn, hidden_size=self.hidden_size_rnn,
                                   num_layers=self.num_layers_rnn, batch_first=True)
            case _:
                print(f"Unsupported recurrent type: {recurrent_type}")

        self.linear = nn.Linear(self.hidden_size_rnn, 1)

    def forward(self, xa, xb):
        batch_size = xa.size(0)

        cnn_outputs = torch.zeros(self.time_steps, batch_size, self.input_size_rnn, device=self.device)

        for time_idx, cnn in enumerate(self.cnn_models):
            xa_t = xa[:, time_idx, :]
            xb_t = xb[:, time_idx, :]

            cnn_outputs[time_idx] = cnn(xa_t, xb_t)

        cnn_outputs = torch.swapaxes(cnn_outputs, 0, 1).to(self.device)

        h0 = torch.zeros(self.num_layers_rnn, batch_size, self.hidden_size_rnn, requires_grad=True,
                         device=self.device).float()
        out, hn = self.rnn(cnn_outputs, h0)

        # out needs to be reshaped into dimensions (batch_size, hidden_size_lin)
        out = nn.functional.tanh(hn)

        # Finally we get out in the shape (batch_size, output_size)
        out = self.linear(out)

        return torch.squeeze(out, 0)

    def cpu(self):
        self.device = "cpu"
        return super().cpu()
