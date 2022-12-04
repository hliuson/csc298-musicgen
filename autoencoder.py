import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from PIL import Image as PILImage
import numpy as np
from evaluate import *

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
    

#Empirically, large kernel sizes seem to do better.
#Since we are dealing iwth 1d data, we can use a large kernel size without much of a performance hit.
class ConvAutoencoder(pl.LightningModule):
    def __init__(self, kernel = 7, encoder_layers = None, decoder_layers = None) -> None:
        super().__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        
        self.threshold = 0.5
        
        # keep output same size as input
        pad = kernel // 2
        if encoder_layers is None:
            encoder_layers = [{
                "in": 128,
                "out": 16,
                "activation": "relu",
                "pool_factor": 1,
            },
            {
                "in": 16,
                "out": 4,
                "activation": "relu",
                "pool_factor": 4,
            }]
        if decoder_layers is None:
            decoder_layers = [{
                "in": 4,
                "out": 16,
                "activation": "relu",
                "upsample_factor": 4,
            },
            {
                "in": 16,
                "out": 128,
                "activation": "sigmoid",
                "upsample_factor": 1,
            }]
        
        for i, layer in enumerate(encoder_layers):
            self.encoder.add_module('conv' + str(i), nn.Conv1d(layer["in"], layer["out"], kernel_size=kernel, padding=pad, groups=1))
            if layer["activation"] == "relu":
                self.encoder.add_module('relu' + str(i), nn.ReLU())
            elif layer["activation"] == "sigmoid":
                self.encoder.add_module('sigmoid' + str(i), nn.Sigmoid())
            if layer["pool_factor"] > 1:
                self.encoder.add_module('pool' + str(i), nn.MaxPool1d(layer["pool_factor"]))
        
        for i, layer in enumerate(decoder_layers):
            self.decoder.add_module('conv' + str(i), nn.Conv1d(layer["in"], layer["out"], kernel_size=kernel, padding=pad, groups=1))
            if layer["activation"] == "relu":
                self.decoder.add_module('relu' + str(i), nn.ReLU())
            elif layer["activation"] == "sigmoid":
                self.decoder.add_module('sigmoid' + str(i), nn.Sigmoid())
            if layer["upsample_factor"] > 1:
                self.decoder.add_module('upsample' + str(i), nn.Upsample(scale_factor=layer["upsample_factor"]))
            
    def forward(self, x):
        x = x.view(x.size(0), 128, -1) #(batch, channels, length)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), -1, 128) #(batch, length, channels)
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
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
