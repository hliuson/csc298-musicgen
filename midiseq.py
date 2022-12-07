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


class MidiFormer(pl.LightningModule):
    def __init__(self, pitch_size=64, dur_size=63, num_layers=4, num_heads=4) -> None:
        super().__init__()
        #save hyperparameters
        self.save_hyperparameters()
        
        self.max_seq_len = 1024
        
        self.pitch_embed = nn.Embedding(128, self.hparams.pitch_size)
        self.dur_embed = nn.Embedding(128, self.hparams.dur_size)
        #expand to (batch, len, 1)
        self.offset_embed = nn.Unflatten(1, (-1, 1))
        
        self.offset_size = 1
        
        #represents special tokens (start, end, pad, mask)
        self.special_tokens = nn.Embedding(100, self.hparams.pitch_size + self.hparams.dur_size)
        #0: mask
        #1: pad
        #2: start
        #3: end
        
        
        self.pitch_head = nn.Sequential(nn.Linear(128, 128), nn.Softmax(dim=2))
        self.dur_head = nn.Sequential(nn.Linear(128, 128), nn.Softmax(dim=2))
        self.offset_head = nn.Sequential(nn.Linear(128, 1), nn.ReLU())
        #sinusoidal positional embedding as a function of time
        #TODO
        self.embed_dim = self.hparams.pitch_size + self.hparams.dur_size + self.offset_size
        
        xformer_config = [
        {
            "reversible": False,  # Turn on to test the effect of using reversible layers
            "block_type": "encoder",
            "num_layers": self.hparams.num_layers,
            "dim_model": self.embed_dim,
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
            "num_layers": self.hparams.num_layers,
            "dim_model": self.embed_dim,
            "residual_norm_style": "pre",
            "position_encoding_config": {
                "name": "sine",
            },
                "multi_head_config_masked": {
                "num_heads": self.hparams.num_heads // 2,
                "residual_dropout": 0.1,
                "attention": {
                    "name": "nystrom",  # whatever attention mechanism
                    "dropout": 0.1,
                    "causal": True,
                    "seq_len": self.max_seq_len,
                },
            },
            "multi_head_config_cross": {
                "num_heads": self.hparams.num_heads // 2,
                "residual_dropout": 0.1,
                "attention": {
                    "name": "nystrom",  # whatever attention mechanism
                    "dropout": 0.1,
                    "causal": True,
                    "seq_len": self.max_seq_len,
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

        
    def training_step(self, batch, batch_idx):
        #x is (batch, length, channels)
        #embed pitch and duration
        
        pitch = self.pitch_embed(batch[:,:,0].long())
        dur = self.dur_embed(batch[:,:,1].long())
        pos = self.offset_embed(batch[:,:,2].long())
        x = torch.cat((pitch, dur), dim=2)
        
        #[0,1] mask for each token in sequence
        mask = (torch.rand((x.shape[0], x.shape[1]), device=self.device) < 0.9).long() #mwhen random number is less than 0.9, keep the note

        #Mask out notes according to mask
        assert len(x.shape) == 3
        x = x*mask.unsqueeze(-1)
        assert len(x.shape) == 3
        #Add embedding of MASK token where mask is 0
        MASK = self.special_tokens(torch.zeros((x.shape[0], x.shape[1]), device=self.device).long())*((1-mask).unsqueeze(-1))
        x = x + MASK
        assert len(x.shape) == 3
        
        x = torch.cat((x, pos), dim=2)
        
        
        y = self.transformer(x)
        #multiclass cross entropy loss
        pitch_loss = F.cross_entropy(self.pitch_head(y), x[:,:,0].long())
        dur_loss = F.cross_entropy(self.dur_head(y), x[:,:,1].long())
        pos_loss = F.mse_loss(self.offset_head(y), x[:,:,2].float())
        loss = pitch_loss + dur_loss + pos_loss
        self.log({'train_loss': loss})
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        #x is (batch, length, channels)
        #embed pitch and duration
        pitch, dur, pos = batch
        
        pitch_emb = self.pitch_embed(pitch.long())
        dur_emb = self.dur_embed(dur.long())
        pos_emb = self.offset_embed(pos.long())
        x = torch.cat((pitch_emb, dur_emb), dim=2)
        
        #[0,1] mask for each token in sequence
        mask = (torch.rand((x.shape[0], x.shape[1]), device=self.device) < 0.9).long() #mwhen random number is less than 0.9, keep the note

        #Mask out notes according to mask
        x = x*mask.unsqueeze(-1)
        #Add embedding of MASK token where mask is 0
        MASK = self.special_tokens(torch.zeros((x.shape[0], x.shape[1]), device=self.device).long())*((1-mask).unsqueeze(-1))
        x = x + MASK
        
        x = torch.cat((x, pos_emb), dim=2)
        
        
        y = self.transformer(x)
        #multiclass cross entropy loss
        pitch_loss = F.cross_entropy(self.pitch_head(y), pitch)
        dur_loss = F.cross_entropy(self.dur_head(y), dur)
        pos_loss = F.mse_loss(self.offset_head(y), pos)
        loss = pitch_loss + dur_loss + pos_loss
        self.log({'val_loss': loss})
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())