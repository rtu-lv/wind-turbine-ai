import torch
import torch.nn as nn
from convolutional_network import ConvolutionalNetwork


class NonStationaryModel(nn.Module):
    def __init__(self, config, num_channels=2, time_steps=20, recurrent_type='RNN'):
        super(NonStationaryModel, self).__init__()

        print("Initializing nonstationary model")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.time_steps = time_steps
        self.recurrent_type = recurrent_type

        self.cnn_models = nn.ModuleList(
            [ConvolutionalNetwork(config, num_channels=num_channels) for i in range(self.time_steps)])

        self.num_layers_recurrent = 1
        self.hidden_size_recurrent = 4
        self.input_size_recurrent = config["cnn_out_features"]

        match self.recurrent_type:
            case "RNN":
                self.rnn = nn.RNN(input_size=self.input_size_recurrent, hidden_size=self.hidden_size_recurrent,
                                  num_layers=self.num_layers_recurrent, batch_first=True)
            case "LSTM":
                self.lstm = nn.LSTM(input_size=self.input_size_recurrent, hidden_size=self.hidden_size_recurrent,
                                    num_layers=self.num_layers_recurrent, batch_first=True)
            case "GRU":
                self.gru = nn.GRU(input_size=self.input_size_recurrent, hidden_size=self.hidden_size_recurrent,
                                  num_layers=self.num_layers_recurrent, batch_first=True)
            case _:
                print(f"Unsupported recurrent type: {recurrent_type}")

        self.linear = nn.Linear(self.hidden_size_recurrent, 1)

    def forward(self, xa, xb):
        batch_size = xa.size(0)

        cnn_outputs = torch.zeros(self.time_steps, batch_size, self.input_size_recurrent, device=self.device)

        for time_idx, cnn in enumerate(self.cnn_models):
            xa_t = xa[:, time_idx, :]
            xb_t = xb[:, time_idx, :]

            cnn_outputs[time_idx] = cnn(xa_t, xb_t)

        cnn_outputs = torch.swapaxes(cnn_outputs, 0, 1).to(self.device)

        h0 = torch.zeros(self.num_layers_recurrent, batch_size, self.hidden_size_recurrent, requires_grad=True,
                         device=self.device).float()

        hn = None
        match self.recurrent_type:
            case "RNN":
                _, hn = self.rnn(cnn_outputs, h0)
            case "LSTM":
                c0 = torch.zeros(self.num_layers_recurrent, batch_size, self.hidden_size_recurrent, requires_grad=True,
                                 device=self.device).float()
                _, (hn, _) = self.lstm(cnn_outputs, (h0, c0))
            case "GRU":
                _, hn = self.gru(cnn_outputs, h0)

        # out needs to be reshaped into dimensions (batch_size, hidden_size_lin)
        # out = nn.functional.tanh(hn)

        # Finally we get out in the shape (batch_size, output_size)
        out = self.linear(hn)

        return torch.squeeze(out, 0)

    def cpu(self):
        self.device = "cpu"
        return super().cpu()
