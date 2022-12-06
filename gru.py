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
    def __init__(self, input_size=128, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.dropout = dropout
        self.loss = nn.MSELoss()
        #Add Batch Normalization layer
        self.norm = nn.BatchNorm1d(input_size)
        self.gru = nn.GRU(input_size, hidden_dim, num_layers=3, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, input_size)
                            , nn.Sigmoid())
    
    def forward(self, x):
        #Our input shape is (batch, sequence length, embedding)
        #Before normalization, we want to reshape the input so that the embedding layer is in axis 1
        x = x.view(x.size(0), x.size(2), x.size(1))
        #Normalize the input
        x = self.norm(x)
        #Reshape the input back to (batch, sequence length, embedding)
        x = x.view(x.size(0), x.size(2), x.size(1))
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

    #Predict the next token in the sequence, based on the whole sequence. Not batched
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pianorolls = batch
        predicted_embeddings = self.forward(pianorolls)
        return predicted_embeddings
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)