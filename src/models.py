import torch
import torch.nn as nn


# ============================================================
# SIMPLE RNN MODEL
# ============================================================

class SimpleRNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):
        super(SimpleRNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        # RNN output
        out, hidden = self.rnn(x)

        # Last time-step output
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)

        return out


# ============================================================
# VANILLA SINGLE-LAYER LSTM
# ============================================================

class VanillaLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):
        super(VanillaLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        # LSTM output
        out, (hidden, cell) = self.lstm(x)

        # Last time-step output
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)

        return out


# ============================================================
# STACKED LSTM
# ============================================================

class StackedLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=2
    ):
        super(StackedLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        # Stacked LSTM output
        out, (hidden, cell) = self.lstm(x)

        # Last time-step output
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)

        return out


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    # Number of input features
    input_size = 13

    # Hidden neurons
    hidden_size = 64

    # Forecast output
    output_size = 1

    # Dummy input:
    # batch = 8
    # sequence length = 24
    # features = 13

    x = torch.randn(
        8,
        24,
        13
    )

    print("Input shape:")
    print(x.shape)


    # --------------------------------------------------------
    # Simple RNN
    # --------------------------------------------------------

    rnn_model = SimpleRNN(
        input_size,
        hidden_size,
        output_size
    )

    rnn_output = rnn_model(x)

    print("\nSimple RNN output:")
    print(rnn_output.shape)


    # --------------------------------------------------------
    # Vanilla LSTM
    # --------------------------------------------------------

    lstm_model = VanillaLSTM(
        input_size,
        hidden_size,
        output_size
    )

    lstm_output = lstm_model(x)

    print("\nVanilla LSTM output:")
    print(lstm_output.shape)


    # --------------------------------------------------------
    # Stacked LSTM
    # --------------------------------------------------------

    stacked_model = StackedLSTM(
        input_size,
        hidden_size,
        output_size,
        num_layers=2
    )

    stacked_output = stacked_model(x)

    print("\nStacked LSTM output:")
    print(stacked_output.shape)


    print("\n===================================")
    print("ALL MODELS TESTED SUCCESSFULLY!")
    print("===================================")