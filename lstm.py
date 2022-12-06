import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from PIL import Image as PILImage
import numpy as np

class autoLSTM(pl.LightningModule):
    def __init__(self, input_size=128, hidden_dim=512, dropout=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.dropout = dropout
        self.loss = nn.MSELoss()
#        self.threshold = 0.5

        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=2, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, input_size)
                            , nn.Sigmoid())

    def forward(self, x):
        #As a recap, the data is of shape (batch, sequence length, embedding_dim) where embedding_dim is 128
        x = x.view(x.size(0), x.size(1), self.input_size) #(batch, sequence length, embedding_dim)
        #Validate that the correct dimensions are being passed in
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        pianorolls = batch
        predicted_embeddings = self.forward(pianorolls[:,:-1,:])
        loss = self.loss(predicted_embeddings, pianorolls[:,1:,:])
        self.log('train_loss', loss)       
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        pianorolls = batch
        predicted_embeddings = self.forward(pianorolls[:,:-1,:])
        loss = self.loss(predicted_embeddings, pianorolls[:,:-1,:])
        self.log('val_loss', loss, sync_dist=True)
#        iou = iou_score(predicted_embeddings, pianorolls[:,1:,:], self.threshold)
#        self.log('val_iou', iou, sync_dist=True)
        return {'loss': loss}

#    def validation_step(self, batch, batch_idx):
#        pianorolls = batch
#        predicted_embeddings = self.forward(pianorolls[:,:-1,:])
#        loss = self.loss(output, pianorolls[:,1:,:])
#        self.log('val_loss', loss, sync_dist=True)
#        return {'loss': loss}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=5e-7, weight_decay=5e-9)

#    def predict(self, batch, batch_idx, dataloader_idx=None):
#        threshold = 0.5
#        batch = pianorolls
#        output = self.forward(pianorolls[:,:-32,:])
#        activation = output > threshold
#        intersection = (activation & pianorolls[:,-32:,:]).sum(dim=2)
#        union = (activation | pianorolls[:,-32:,:]).sum(dim=2)
#        if (union == 0).any():
#            union[union==0] = 1
#        iou = intersection / union
#        return {'iou': iou}


