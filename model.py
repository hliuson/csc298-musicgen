import torch.nn as nn
import torch
### LSTM Model with fully connected output head which predicts the next 32 steps of a pianoroll.
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        _, hc = self.lstm(x)
        z = None
        for i in range(32):
            zero = torch.zeros(x.shape[0], 1, 128)
            output, hc = self.lstm(zero, hc)
            output = self.fc(output)
            z = torch.cat([x, output], dim=1)
        return z

def get_baseline_model():
    return LSTMModel()