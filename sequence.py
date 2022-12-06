import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from PIL import Image as PILImage
import numpy as np
from evaluate import *
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
                    "causal": True,
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
        output = self.forward(batch)
        loss = F.mse_loss(output, batch)
        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = F.mse_loss(output, batch)
        self.log('val_loss', loss, sync_dist=True)
        
        #cosine similarity
        sim = self.cos(output, batch)
        self.log('val_cos', sim.mean(), sync_dist=True)
        return {'loss': loss}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer
        