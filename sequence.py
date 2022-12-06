import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from PIL import Image as PILImage
import numpy as np
from autoencoder import *
from infer import *
from xformers.factory.model_factory import *

# Small Transformer model
class TransformerSequence(pl.LightningModule):
    def __init__(self, num_heads=4, num_layers=3, embedding_size=128, hidden_size=256) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Autoregressive Transformer for next-step prediction
        # Inputs/outputs are (batch, seq_len, channels).
    
        
        #small transformer with nystrom attention, and otherwise standard transformer
        xformer_config = [
        {
            "reversible": False,  # Turn on to test the effect of using reversible layers
            "block_type": "encoder",
            "num_layers": num_layers,
            "dim_model": embedding_size,
            "residual_norm_style": "pre",
            "position_encoding_config": {
                "name": "sine",
            },
            "multi_head_config": {
                "num_heads": num_heads,
                "residual_dropout": 0.1,
                "use_rotary_embeddings": True,
                "attention": {
                    "name": "nystrom",
                    "dropout": 0.1,
                    "causal": True, #causal attention means that the model can only see the past
                },
            },
            "feedforward_config": {
                "name": "MLP",  
                "dropout": 0.1  ,
                "activation": "gelu",
                "hidden_layer_multiplier": 2, # hidden layer size is 4x the embedding size. May want to decrease this. 
            },
        }
        ]
        config = xFormerConfig(xformer_config)
        self.transformer = xFormer.from_config(config)
        
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        
    def forward(self, x):
        return self.transformer(x) #shape of batch is (batch_size, seq_len, channels)
    
    def training_step(self, batch, batch_idx):
        
        output = self.forward(batch[: , :-1, :])
        loss = F.mse_loss(output, batch[:, 1:, :])
        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        output = self.forward(batch[: , :-1, :])
        loss = F.mse_loss(output, batch[:, 1:, :])
        self.log('val_loss', loss, sync_dist=True)
        
        #cosine similarity
        sim = self.cos(output, batch[:, 1:, :])
        self.log('val_cos', sim.mean(), sync_dist=True)
        return {'loss': loss}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer
    
class NoteTransformer(pl.LightningModule):
    def __init__(self, num_heads=4, num_layers=3, embedding_size=128, hidden_size=256) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Autoregressive Transformer for next-step prediction
        # Inputs/outputs are (batch, seq_len, channels).
    
        
        #small transformer with nystrom attention, and otherwise standard transformer
        xformer_config = [
        {
            "reversible": False,  # Turn on to test the effect of using reversible layers
            "block_type": "encoder",
            "num_layers": num_layers,
            "dim_model": embedding_size,
            "residual_norm_style": "pre",
            "position_encoding_config": {
                "name": "sine",
            },
            "multi_head_config": {
                "num_heads": num_heads,
                "residual_dropout": 0.1,
                "use_rotary_embeddings": True,
                "attention": {
                    "name": "nystrom",
                    "dropout": 0.1,
                    "causal": False,
                },
            },
            "feedforward_config": {
                "name": "MLP",  
                "dropout": 0.1  ,
                "activation": "gelu",
                "hidden_layer_multiplier": 2, # hidden layer size is 4x the embedding size. May want to decrease this. 
            },
        },
        {
            "reversible": False,  # Turn on to test the effect of using reversible layers
            "block_type": "decoder",
            "num_layers": num_layers,
            "dim_model": embedding_size,
            "residual_norm_style": "pre",
            "position_encoding_config": {
                "name": "sine",
            },
            "multi_head_config": {
                "num_heads": num_heads,
                "residual_dropout": 0.1,
                "use_rotary_embeddings": True,
                "attention": {
                    "name": "nystrom",
                    "dropout": 0.1,
                    "causal": True, #autoregressive decoder
                },
            },
            "feedforward_config": {
                "name": "MLP",  
                "dropout": 0.1  ,
                "activation": "gelu",
                "hidden_layer_multiplier": 2, # hidden layer size is 4x the embedding size. May want to decrease this. 
            },
        }
        ]
        config = xFormerConfig(xformer_config)
        self.transformer = xFormer.from_config(config)

        
    def forward(self, x):
        return self.transformer(x) #shape of batch is (batch_size, seq_len, channels)
    
    def training_step(self, batch, batch_idx):
        
        output = self.forward(batch[: , :-1, :])
        loss = F.cross_entropy(output, batch[:, 1:, :])
        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        output = self.forward(batch[: , :-1, :])
        loss = F.cross_entropy(output, batch[:, 1:, :])
        self.log('val_loss', loss, sync_dist=True)
        
        iou = iou_score(output, batch[:, 1:, :], 0.5)
        self.log('val_iou', iou, sync_dist=True)
        return {'loss': loss}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer
        
class Autoformer(pl.LightningModule):  
    def __init__(self, num_heads=4, num_layers=5, embedding_size=64, phrase_len=32) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Autoregressive Transformer for next-step prediction
        # Inputs/outputs are (batch, seq_len, channels).
    
        
        #small transformer with nystrom attention, and otherwise standard transformer
        xformer_config = [
        {
            "reversible": False,  # Turn on to test the effect of using reversible layers
            "block_type": "encoder",
            "num_layers": self.hparams.num_layers,
            "dim_model": self.hparams.embedding_size,
            "residual_norm_style": "pre",
            "position_encoding_config": {
                "name": "sine",
            },
            "multi_head_config": {
                "num_heads": self.hparams.num_heads,
                "residual_dropout": 0.1,
                "use_rotary_embeddings": True,
                "attention": {
                    "name": "nystrom",
                    "dropout": 0.1,
                    "causal": True, #causal attention means that the model can only see the past
                },
            },
            "feedforward_config": {
                "name": "MLP",  
                "dropout": 0.1  ,
                "activation": "gelu",
                "hidden_layer_multiplier": 4, # hidden layer size is 4x the embedding size. May want to decrease this. 
            },
        }
        ]
        config = xFormerConfig(xformer_config)
        self.transformer = xFormer.from_config(config)
        self.autoencoder = Autoformer_Autoencoder(conv_dim=16, embed_dim=self.hparams.embedding_size, seq_length=self.hparams.phrase_len)

        
    def forward(self, x):
        embed = self.autoencoder.encode(x).reshape(x.shape[0], -1, self.hparams.embedding_size)
        output = self.transformer(embed).reshape(-1, self.hparams.embedding_size) #stack embeddings and pass through transformer
        logits = self.autoencoder.decode(output).reshape(x.shape[0], -1, 128) #reshape to match batch shape
        return logits #shape of logits is (batch_size, seq_len, channels)
    
    def training_step(self, batch, batch_idx): #batch is (batch_size, seq_len, 128)
        #if batch has length not a multiple of 32, pad it with zeros, also convert to float
        batch = F.pad(batch, (0, 0, 0, self.hparams.phrase_len - batch.shape[1] % self.hparams.phrase_len))
        batch = batch.float()
        logits = self.forward(batch)
        #remove first phrase_len elements of batch and last phrase_len elements of logits
        #Essentially left shift the batch by phrase_len and compare to logits.
        loss = F.cross_entropy(logits[:, :-self.hparams.phrase_len, :], batch[:, self.hparams.phrase_len:, :])
        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        batch = F.pad(batch, (0, 0, 0, self.hparams.phrase_len - batch.shape[1] % self.hparams.phrase_len))
        batch = batch.float()
        logits = self.forward(batch)
        #ensure shape is correct for cross entropy loss, and shift batch and logits
        print(logits.shape, batch.shape)
        
        loss = F.cross_entropy(logits[:, :-self.hparams.phrase_len, :], batch[:, self.hparams.phrase_len:, :])
        
        self.log('val_loss', loss, sync_dist=True)
        
        iou = iou_score(logits[:, :-self.hparams.phrase_len, :], batch[:, self.hparams.phrase_len:, :], 0.5)
        self.log('val_iou', iou, sync_dist=True)
        return {'loss': loss}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer   

class Autoformer_Autoencoder(pl.LightningModule):
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
        x = x.view(-1, 128, self.seq_length) #(batch, channels, length)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, self.seq_length, 128) #(batch, length, channels)
        return x

    def encode(self, x):    
        x = x.view(-1, 128, self.seq_length).float() #(batch, channels, length)
        x = self.encoder(x)
        x = x.view(-1, self.embed_dim) #(batch, channels)
        return x
    
    def decode(self, x):
        x = x.view(-1, self.embed_dim) #(batch, channels)
        x = self.decoder(x)
        x = x.view(-1, self.seq_length, 128) #(batch, length, channels)
        return x
    
def iou_score(raw, truth, threshold):
    #shape of raw and truth is (batch_size, 32, 128)
    activation = raw > threshold
    truth = truth > 0.5
    intersectionMap = torch.logical_and(activation, truth)
    unionMap = torch.logical_or(activation, truth)
    
    intersections = intersectionMap.sum(dim=(1, 2)) #shape (batch_size)
    unions = unionMap.sum(dim=(1, 2))

    iou = intersections / unions
    #impute 1 for NaNs, because if union is 0 then there are no ground truth positives
    iou = torch.where(torch.isnan(iou), torch.ones_like(iou), iou)
    return iou.mean()