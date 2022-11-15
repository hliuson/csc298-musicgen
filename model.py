import torch.nn as nn
import torch
import pytorch_lightning as pl

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

class LighningLSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = LSTMModel()
        self.loss = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pianorolls = batch
        outputs= self.model(pianorolls[:, :-32, :])
        # Get the rest of the 32 steps
        
        
        # Compute loss against the last 32 steps
        loss = self.loss(
            outputs,
            pianorolls[:, -32:, :],
        )
        

        train_loss += loss.item()  * pianorolls.shape[0]
        #this is an instance of multi-label classification, so we need some threshold to determine if a note is on or off
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        pianorolls = batch
        outputs= self.model(pianorolls[:, :-32, :])
        # Get the rest of the 32 steps
                
                
        # Compute loss against the last 32 steps
        loss = self.loss(
            outputs,
            pianorolls[:, -32:, :],
        )
        threshold = 0.5
        activation = outputs > threshold
        pianorolls = pianorolls.bool()
        #Define accuracy as intersection over union of the predicted and actual notes per timestep
        intersection = (activation & pianorolls[:, -32:, :]).sum(dim=2)
        union = (activation | pianorolls[:, -32:, :]).sum(dim=2)
        # if union is 0, then the intersection is also 0, so we can just set the union to 1
        if (union == 0).any():
            union[union == 0] = 1
        return {'val_loss': loss, 'iou': intersection / union}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)