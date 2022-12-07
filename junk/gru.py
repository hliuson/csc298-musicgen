#Train a GRU to act on the same sequences as lstm.py

import torch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from PIL import Image as PILImage
import numpy as np

class autoGRU(pl.LightningModule):
    def __init__(self, input_size=128, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.dropout = dropout
        self.loss = nn.MSELoss()
        self.gru = nn.GRU(input_size, hidden_dim, num_layers=3, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, input_size)
                            , nn.Sigmoid())
    
    def forward(self, x):
        #Normalize x by axis 2. Subtract the mean and divide by the standard deviation
        x, _ = self.gru(x)
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        pianorolls = batch
        pianorolls = (pianorolls - pianorolls.mean(axis=2, keepdim=True)) / pianorolls.std(axis=2, keepdim=True)
        predicted_embeddings = self.forward(pianorolls[:,:-1,:])
        loss = self.loss(predicted_embeddings, pianorolls[:,1:,:])
        self.log('train_loss', loss)       
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        pianorolls = batch
        pianorolls = (pianorolls - pianorolls.mean(axis=2, keepdim=True)) / pianorolls.std(axis=2, keepdim=True)
        predicted_embeddings = self.forward(pianorolls[:,:-1,:])
        loss = self.loss(predicted_embeddings, pianorolls[:,1:,:])
        self.log('val_loss', loss, sync_dist=True)
        return {'loss': loss}

    #Predict the next token in the sequence, based on the whole sequence. Not batched
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pianorolls = batch
        predicted_embeddings = self.forward(pianorolls)
        return predicted_embeddings
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)