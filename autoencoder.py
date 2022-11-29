import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from PIL import Image as PILImage
import numpy as np
from evaluate import *


### Autoencoder for a 32-step pianoroll (shape: (batch_size, 32, 128))
### Encoder: Convolutional -> Fully Connected
### Decoder: Fully Connected -> Convolutional
### We use the convolutional layers to extract local ideas

class ConvAutoEncoder(torch.nn.Module):
    def __init__(self, conv_depth = 2, fc_depth = 1, conv_width = 32, fc_width=1024,
                 encoded_length=512, dropout=0.5): 
        super().__init__()
        self.encodeConv = torch.nn.Sequential()
        self.encodeFc = torch.nn.Sequential()
        self.decoderConv = torch.nn.Sequential()
        self.decoderFc = torch.nn.Sequential()
        
        self.poolFactor = 1#2**conv_depth

    
        for i in range(conv_depth):
            in_channels = conv_width
            out_channels = conv_width
            if i == 0:
                in_channels = 128
            self.encodeConv.add_module(f'conv_{i}', torch.nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2))
            #self.encoder.add_module(f'batchnorm_{i}', torch.nn.BatchNorm1d(conv_width))
            self.encodeConv.add_module(f'relu_{i}', torch.nn.ReLU())
            self.encodeConv.add_module(f'dropout_{i}', torch.nn.Dropout(dropout))
            #self.encodeConv.add_module(f'pool_{i}', torch.nn.MaxPool1d(kernel_size=2))
            
        for i in range(fc_depth):
            in_dim = fc_width
            out_dim = fc_width
            if i == 0:
                in_dim = 32*conv_width // self.poolFactor
            if i == fc_depth - 1:
                out_dim = encoded_length
            self.encodeFc.add_module(f'fc_{i}', LinearSkipBlock(in_dim, out_dim))
            
        for i in range(fc_depth):
            in_dim = fc_width
            out_dim = fc_width
            if i == 0:
                in_dim = encoded_length
            if i == fc_depth - 1:
                out_dim = 32*conv_width // self.poolFactor
            self.decoderFc.add_module(f'fc_{i}', LinearSkipBlock(in_dim, out_dim))
            
        for i in range(conv_depth):
            in_channels = conv_width
            out_channels = conv_width
            if i == conv_depth - 1:
                out_channels = 128
            self.decoderConv.add_module(f'conv_{i}', torch.nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2))
            self.decoderConv.add_module(f'relu_{i}', torch.nn.ReLU())
            # we should not apply dropout to the last layer
            if i != conv_depth - 1:
                self.decoderConv.add_module(f'dropout_{i}', torch.nn.Dropout(dropout))
            #self.decoderConv.add_module(f'pool_{i}', torch.nn.ConvTranspose1d(in_channels=conv_width, out_channels=conv_width, kernel_size=2, stride=2))
        
    def forward(self, x):
        #Conv1d expects data in format (batch, channels, length)
        x = x.view(x.size(0), 128, -1)
        x = self.encodeConv(x)
        x = x.view(x.size(0), -1)
        x = self.encodeFc(x)
        x = self.decoderFc(x)
        x = x.view(x.size(0), -1, 32 // self.poolFactor)
        x = self.decoderConv(x)
        x = x.view(x.size(0), -1, 128)
        return x
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

class LinearSkipBlock(torch.nn.Module):
    def __init__(self, x_dim, y_dim, dropout=0.5):
        super().__init__()
        self.indim = x_dim
        self.outdim = y_dim
        self.linear = torch.nn.Linear(x_dim, y_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        z = self.linear(x)
        z = self.relu(z)
        if self.indim == self.outdim:
            z = z + x
        z = self.dropout(z)
        return z

class LinearBlock(torch.nn.Module):
    def __init__(self, x_dim, y_dim, dropout=0.5):
        super().__init__()
        self.linear = torch.nn.Linear(x_dim, y_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class LightningConvAutoencoder(pl.LightningModule):
    def __init__(self, model = None):
        super().__init__()
        if model is None:
            model = ConvAutoEncoder(fc_width=128, encoded_length=128, fc_depth=5, conv_depth=3, dropout=0.5)
        self.model = model
        self.loss = torch.nn.MSELoss()
        
    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = self.loss(output, batch)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = self.loss(output, batch)
        self.log('val_loss', loss, sync_dist=True)
        
        intersection = torch.sum(output * batch)
        union = torch.sum(output) + torch.sum(batch) - intersection
        iou = intersection / union
        self.log('val_iou', iou, sync_dist=True)
        return {'loss': loss, 'iou': iou}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
    
    
#Use conv1d to reduce the dimensions of the input,
#then MLP layers to encode the data
class ReconstructLossAutoencoder(pl.LightningModule):
    def __init__(self, conv_embed_dim=16, fc_widths=[256, 256], encoded_length=128, dropout=0.5, chunk_length=32, inner_loss_weight=0):
        super().__init__()
        self.chunk_length = chunk_length
        self.inner_loss_weight = inner_loss_weight
        self.inner_loss = torch.nn.MSELoss()
        self.outer_loss = torch.nn.BCELoss()
        self.encoderconv = torch.nn.Sequential()
        self.encoderconv.add_module('conv1', torch.nn.Conv1d(128, conv_embed_dim, kernel_size=1, padding=0))
        self.encoderconv.add_module('relu1', torch.nn.ReLU())
        self.decoderconv = torch.nn.Sequential()
        self.decoderconv.add_module('conv1', torch.nn.Conv1d(conv_embed_dim, 128, kernel_size=1, padding=0))
        self.decoderconv.add_module('relu1', torch.nn.ReLU())
        self.encoderfc = torch.nn.ModuleList()
        self.decoderfc = torch.nn.ModuleList()
        
        self.sig = torch.nn.Sigmoid()
        
        self.threshold = 0.5
        
        finaldim = conv_embed_dim*chunk_length
        
        fc_widths.append(encoded_length)
        fc_widths.insert(0, finaldim)
        
        # finaldim -> fc_widths[0] -> ... -> fc_widths[n] -> encoded_length
        for i in range(len(fc_widths)-1):
            in_dim = fc_widths[i]
            out_dim = fc_widths[i+1]
            self.encoderfc.add_module(f'linear_{i}', LinearBlock(in_dim, out_dim, dropout))
    
        # encoded_length -> fc_widths[n] -> ... -> fc_widths[0] -> finaldim
        for i in range(len(fc_widths)-1):
            in_dim = fc_widths[-i-1]
            out_dim = fc_widths[-i-2]
            self.decoderfc.add_module(f'linear_{i}', LinearBlock(in_dim, out_dim, dropout))
    
    def forward(self, x):
        forwards = []
        backwards = []
        
        x = x.view(x.size(0), 128, -1) #(batch, channels, length)
        x = self.encoderconv(x)
        x = x.view(x.size(0), -1) #(batch, channels*length)
        for layer in self.encoderfc:
            forwards.append(x)
            x = layer(x)
        for layer in self.decoderfc:
            x = layer(x)
            backwards.append(x)
        x = x.view(x.size(0), -1, self.chunk_length) #(batch, channels, length)
        x = self.decoderconv(x)
        x = x.view(x.size(0), -1, 128) #(batch, length, channels)
        x = self.sig(x)
        return x, forwards, backwards
    
    def training_step(self, batch, batch_idx):
        output, fwds, bwds = self(batch)
        loss = self.outer_loss(output, batch)
        
        inner_loss = 0
        #Add the loss of the forward and backward layers
        for i in range(len(fwds)):
            inner_loss += self.inner_loss(fwds[i], bwds[-i-1]) * self.inner_loss_weight
        self.log('train_loss', loss)
        self.log('train_inner_loss', inner_loss)
        # inner loss and outer loss have different number of parameters so we need to weight them
        total = loss + inner_loss
        
        
        
        self.log('train_total_loss', total)
        return {'loss': total}
    
    def validation_step(self, batch, batch_idx):
        output, fwds, bwds = self(batch)
        loss = self.outer_loss(output, batch)
        
        activation = output > self.threshold
        #logical and
        intersection = torch.sum(activation * batch)

        union = torch.sum(activation) + torch.sum(batch) - intersection
        iou = intersection / union
        #normalize by number of gpus
        iou = iou / torch.distributed.get_world_size()
        self.log('val_iou', iou, on_epoch=True, sync_dist=True) 
        
        inner_loss = 0
        #Add the loss of the forward and backward layers
        for i in range(len(fwds)):
            inner_loss += self.inner_loss(fwds[i], bwds[-i-1]) * self.inner_loss_weight
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_inner_loss', inner_loss, sync_dist=True)
        total = loss + inner_loss
        self.log('val_total_loss', total, sync_dist=True)
        
        if batch_idx == 0:
            base = batch[0].cpu().detach().numpy() * 255
            base = PILImage.fromarray(base.astype(np.uint8), mode='L')
            reconstruction = activation[0].cpu().detach().numpy() * 255
            reconstruction = PILImage.fromarray(reconstruction.astype(np.uint8), mode='L')

            self.logger.log_image(key='val_images', images=[wandb.Image(base)], caption=['base'])
            self.logger.log_image(key='val_images', images=[wandb.Image(reconstruction)], caption=['reconstruction'])
            
            
        return {'loss': total, 'iou': iou}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        
class SimpleAutoencoder(pl.LightningModule):
    def __init__(self, conv_dim = 16, kernel = 3) -> None:
        super().__init__()
        self.model = nn.Sequential()
        
        # keep output same size as input
        pad = kernel // 2
        
        self.threshold = 0.5
        
        self.model.add_module('conv1', nn.Conv1d(128, conv_dim, kernel_size=kernel, padding=pad, groups=1))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('conv2', nn.Conv1d(conv_dim, 128, kernel_size=kernel, padding=pad))
        self.model.add_module('sigmoid', nn.Sigmoid())
    
    def forward(self, x):
        x = x.view(x.size(0), 128, -1) #(batch, channels, length)
        x = self.model(x)
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
        return {'loss': loss}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
    
    def encode(self, x):
        return self.model[:2](x)
    
    def decode(self, x):
        return self.model[2:](x)
