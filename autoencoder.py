import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from PIL import Image as PILImage
import numpy as np
from evaluate import *

#best
def load_autoencoder():
    #fetch best model from checkpoint using pytorch lightning
    folder = "checkpoints/autoencoder-simple-mlp-dropout0.2-8-13-128/"
    checkpoint = "epoch=99-step=9600.ckpt"
    return SimpleAutoencoderMLP.load_from_checkpoint(folder+checkpoint)

#Compresses data by a factor of 128/conv_dim. For conv_dim=16, this is 8x compression.
class SimpleAutoencoder(pl.LightningModule):
    def __init__(self, conv_dim = 16, kernel = 3, seq_length=32) -> None:
        super().__init__()
        self.model = nn.Sequential()
        
        # keep output same size as input
        pad = kernel // 2
        
        self.threshold = 0.5
        self.conv_dim = conv_dim
        self.seq_length = seq_length
        
        self.model.add_module('conv1', nn.Conv1d(128, conv_dim, kernel_size=kernel, padding=pad, groups=1))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('conv2', nn.Conv1d(conv_dim, 128, kernel_size=kernel, padding=pad))
        self.model.add_module('sigmoid', nn.Sigmoid())
    
    def forward(self, x):
        x = x.view(x.size(0), 128, self.seq_length) #(batch, channels, length)
        x = self.model(x)
        x = x.view(x.size(0), self.seq_length, 128) #(batch, length, channels)
        return x

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = F.binary_cross_entropy(output, batch)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = F.binary_cross_entropy(output, batch)
        self.log('val_loss', loss, sync_dist=True)
        iou = iou_score(output, batch, self.threshold)
        self.log('val_iou', iou, sync_dist=True)
        return {'loss': loss}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
    
    def encode(self, x):    
        x = x.view(x.size(0), 128, self.seq_length).float() #(batch, channels, length)
        x = self.model[:2](x)
        x = x.view(x.size(0), self.seq_length*self.conv_dim) #(batch, channels)
        
        return x
    
    def decode(self, x):
        x = x.view(x.size(0), self.conv_dim, self.seq_length) #(batch, channels, length)
        x = self.model[2:](x)
        x = x.view(x.size(0), self.seq_length, 128) #(batch, length, channels)
        return x
    
class SimpleAutoencoderMLP(pl.LightningModule):
    def __init__(self, conv_dim = 8, kernel = 13, seq_length=32, embed_dim = 128) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        
        # keep output same size as input
        pad = kernel // 2
        
        self.threshold = 0.5
        self.conv_dim = conv_dim
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        
        self.encoder.add_module('conv1', nn.Conv1d(128, conv_dim, kernel_size=kernel, padding=pad, groups=1))
        self.encoder.add_module('relu1', nn.ReLU())
        self.encoder.add_module('drop2', nn.Dropout(0.2))
        self.encoder.add_module('flatten', nn.Flatten())
        self.encoder.add_module('fc1', nn.Linear(self.seq_length*conv_dim, embed_dim))
        self.encoder.add_module('relu2', nn.ReLU())
        self.encoder.add_module('drop1', nn.Dropout(0.2))
        
        self.decoder.add_module('fc2', nn.Linear(embed_dim, self.seq_length*conv_dim))
        self.decoder.add_module('relu3', nn.ReLU())
        self.decoder.add_module('drop3', nn.Dropout(0.2))
        self.decoder.add_module('unflatten', nn.Unflatten(1, (conv_dim, self.seq_length))) #unflatten to (batch, channels, length)
        self.decoder.add_module('conv2', nn.Conv1d(conv_dim, 128, kernel_size=kernel, padding=pad))
        self.decoder.add_module('sigmoid', nn.Sigmoid())
    
    def forward(self, x):
        x = x.view(x.size(0), 128, self.seq_length) #(batch, channels, length)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), self.seq_length, 128) #(batch, length, channels)
        return x

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = F.binary_cross_entropy(output, batch)
        #penalize too many activations in the same time step by summing and squaring the activations
        penalty = torch.sum(torch.sum(output, dim=2)**2) * 1e-2
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = F.binary_cross_entropy(output, batch)
        self.log('val_loss', loss, sync_dist=True)
        iou = iou_score(output, batch, self.threshold)
        self.log('val_iou', iou, sync_dist=True)
        return {'loss': loss}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
    
    def encode(self, x):    
        x = x.view(x.size(0), 128, self.seq_length).float() #(batch, channels, length)
        x = self.encoder(x)
        x = x.view(x.size(0), self.embed_dim) #(batch, channels)
        return x
    
    def decode(self, x):
        x = x.view(x.size(0), self.embed_dim) #(batch, channels)
        x = self.decoder(x)
        x = x.view(x.size(0), self.seq_length, 128) #(batch, length, channels)
        return x