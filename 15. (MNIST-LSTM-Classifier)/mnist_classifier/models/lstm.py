from unicodedata import bidirectional
import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=4, dropout_p=2):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            # [ batch_first ]
            # default = False => 입력 : (length, batch_size, input_size) / 출력 : (length, batch_size, hidden_size * 2)
            # True => 입력 : (batch_size, length, input_size) / 출력 : (batch_size, length, hidden_size * 2)
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True,  # 양방향 여부
        )

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),          # 양방향 => hidden_size * 2
            nn.Linear(hidden_size * 2, output_size),  # 양방향 => hidden_size * 2
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # |x| => (batch_size, h, w) : 여기서는 N(시퀀스 길이 => h)

        z, _ = self.lstm(x)
        # |z| => (batch_size, h, hidden_size * 2)
        z = z[:, -1]  # 마지막 가져오기 (분류니까)
        # |z| => (batch_size, hidden_size * 2)

        y = self.layers(z)
        # |y| => (Batch_size, output_size)

        return y
