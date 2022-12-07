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
    def __init__(self, num_layers=4, num_heads=4, pitch_size=32, dur_size=32, offset_size=32, time_num_size=16, time_denom_size=16,
                 bar_size = 32, inst_size = 32, tmp_size = 32, vel_size=32, model_dim = 128) -> None:
        super().__init__()
        #save hyperparameters
        self.save_hyperparameters()
        
        self.max_seq_len = 1024
        
        self.pitch_embed = nn.Embedding(128, self.hparams.pitch_size)
        self.dur_embed = nn.Embedding(128, self.hparams.dur_size)
        self.offset_embed = nn.Embedding(64, self.hparams.offset_size)
        self.time_num_embed = nn.Embedding(32, self.hparams.time_num_size)
        self.time_denom_embed = nn.Embedding(32, self.hparams.time_denom_size)
        self.bar_embed = nn.Embedding(512, self.hparams.bar_size)
        self.inst_embed = nn.Embedding(128, self.hparams.inst_size)
        self.tmp_embed = nn.Embedding(32, self.hparams.tmp_size)
        self.vel_embed = nn.Embedding(32, self.hparams.vel_size)
        
        self.mixer = nn.Sequential(nn.Linear(self.hparams.pitch_size + self.hparams.dur_size + self.hparams.offset_size + self.hparams.time_num_size + self.hparams.time_denom_size
                                             + self.hparams.bar_size + self.hparams.inst_size + self.hparams.tmp_size + self.hparams.vel_size, self.hparams.model_dim), nn.ReLU())
        
        #represents special tokens (start, end, pad, mask)
        self.special_tokens = nn.Embedding(4, self.hparams.pitch_size + self.hparams.dur_size)
        #0: mask
        #1: pad
        #2: start
        #3: end
        
        
        self.pitch_head = nn.Sequential(nn.Linear(128, 128), nn.Softmax(dim=2))
        self.dur_head = nn.Sequential(nn.Linear(128, 128), nn.Softmax(dim=2))
        self.offset_head = nn.Sequential(nn.Linear(128, 64), nn.Softmax(dim=2))
        self.time_num_head = nn.Sequential(nn.Linear(128, 32), nn.Softmax(dim=2))
        self.time_denom_head = nn.Sequential(nn.Linear(128, 32), nn.Softmax(dim=2))
        self.bar_head = nn.Sequential(nn.Linear(128, 512), nn.Softmax(dim=2))
        self.inst_head = nn.Sequential(nn.Linear(128, 128), nn.Softmax(dim=2))
        self.tmp_head = nn.Sequential(nn.Linear(128, 32), nn.Softmax(dim=2))
        self.vel_head = nn.Sequential(nn.Linear(128, 32), nn.Softmax(dim=2))
        
        xformer_config = [
        {
            "reversible": False,  # Turn on to test the effect of using reversible layers
            "block_type": "encoder",
            "num_layers": self.hparams.num_layers,
            "dim_model": self.hparam.model_dim,
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
            "dim_model": self.hparam.model_dim,
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
        loss = self.BERT_step(batch)
        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.BERT_step(batch)
        self.log('val_loss', loss, sync_dist=True)
        return {'loss': loss}

    def BERT_step(self, batch):
        #x is (batch, length, channels)
        #embed pitch and duration
        pitch, dur, off, tden, tnum, bar, inst, tmp, vel = batch
        
        x = torch.cat((self.pitch_embed(pitch),
                       self.dur_embed(dur),
                       self.offset_embed(off),
                       self.time_denom_embed(tden),
                       self.time_num_embed(tnum),
                       self.bar_embed(bar),
                       self.inst_embed(inst),
                       self.tmp_embed(tmp),
                       self.vel_embed(vel)), dim=2)
        
        #[0,1] mask for each token in sequence
        mask = (torch.rand((x.shape[0], x.shape[1]), device=self.device) < 0.8).long() #mwhen random number is less than 0.8, keep the note

        #Mask out notes according to mask
        x = x*mask.unsqueeze(-1)
        #Add embedding of MASK token where mask is 0
        MASK = self.special_tokens(torch.zeros((x.shape[0], x.shape[1]), device=self.device).long())*((1-mask).unsqueeze(-1))
        x = x + MASK
        
        y = self.transformer(x)
        #multiclass cross entropy loss
        pitch_loss = F.cross_entropy(self.pitch_head(y), F.one_hot(pitch.long(), num_classes=128).float())
        dur_loss = F.cross_entropy(self.dur_head(y), F.one_hot(dur.long(), num_classes=128).float())
        offset_loss = F.cross_entropy(self.offset_head(y), F.one_hot(off.long(), num_classes=128).float())
        tden_loss = F.cross_entropy(self.time_denom_head(y), F.one_hot(tden.long(), num_classes=32).float())
        tnum_loss = F.cross_entropy(self.time_num_head(y), F.one_hot(tnum.long(), num_classes=32).float())
        bar_loss = F.cross_entropy(self.bar_head(y), F.one_hot(bar.long(), num_classes=512).float())
        inst_loss = F.cross_entropy(self.inst_head(y), F.one_hot(inst.long(), num_classes=128).float())
        tmp_loss = F.cross_entropy(self.tmp_head(y), F.one_hot(tmp.long(), num_classes=32).float())
        vel_loss = F.cross_entropy(self.vel_head(y), F.one_hot(vel.long(), num_classes=32).float())
        loss = pitch_loss + dur_loss + offset_loss + tden_loss + tnum_loss + bar_loss + inst_loss + tmp_loss + vel_loss
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())