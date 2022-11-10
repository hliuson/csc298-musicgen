import torch.nn as nn
import torch
### LSTM Model with fully connected output head which predicts the next 32 steps of a pianoroll.
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=512,
            proj_size=128,
            num_layers=6,
            batch_first=True,
            dropout=0.5,
        )
        self.fc = nn.Linear(128, 128)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, hc = self.lstm(x)
        z = None
        #get device 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(32):
            zero = torch.zeros(x.shape[0], 1, 128, device=device)
            output, hc = self.lstm(zero, hc)
            output = self.sigmoid(output)
            output = self.fc(output).reshape(-1, 1, 128)
            if z is None:
                z = output
            else:
                z = torch.cat((z, output), dim=1)
        return z

def get_baseline_model():
    return LSTMModel()