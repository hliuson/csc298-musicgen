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
    def __init__(self, input_size, hidden_dim, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.loss = nn.MSELoss()

        self.gru = nn.GRU(input_size, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, input_size)
                            , nn.Sigmoid())
    
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), self.input_size) #(batch, sequence length, embedding_dim)
        x, _ = self.gru(x)
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
        loss = self.loss(predicted_embeddings, pianorolls[:,1:,:])
        self.log('val_loss', loss, sync_dist=True)
        return {'loss': loss}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-7)