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
                 bar_size = 32, inst_size = 32, tmp_size = 32, vel_size=32, model_dim = 128, hidden_layer_multiplier=4) -> None:
        super().__init__()
        #save hyperparameters
        self.save_hyperparameters()
        
        self.max_seq_len = 1024
        
        self.pitch_embed = nn.Embedding(128, self.hparams.pitch_size)
        self.dur_embed = nn.Embedding(128, self.hparams.dur_size)
        self.offset_embed = nn.Embedding(64, self.hparams.offset_size)
        self.time_num_embed = nn.Embedding(32, self.hparams.time_num_size)
        self.time_denom_embed = nn.Embedding(32, self.hparams.time_denom_size)
        self.bar_embed = nn.Embedding(1024, self.hparams.bar_size)
        self.inst_embed = nn.Embedding(128, self.hparams.inst_size)
        self.tmp_embed = nn.Embedding(32, self.hparams.tmp_size)
        self.vel_embed = nn.Embedding(32, self.hparams.vel_size)
        
        self.mixer = nn.Sequential(nn.Linear(self.hparams.pitch_size + self.hparams.dur_size + self.hparams.offset_size + self.hparams.time_num_size + self.hparams.time_denom_size
                                             + self.hparams.bar_size + self.hparams.inst_size + self.hparams.tmp_size + self.hparams.vel_size, self.hparams.model_dim), nn.ReLU())
        
        #represents special tokens (pad, mask)
        self.special_tokens = nn.Embedding(2, self.hparams.model_dim)
        #0: mask
        #1: pad
        
        
        self.pitch_head = nn.Sequential(nn.Linear(self.hparams.model_dim, 128), nn.Softmax(dim=2))
        self.dur_head = nn.Sequential(nn.Linear(self.hparams.model_dim, 128), nn.Softmax(dim=2))
        self.offset_head = nn.Sequential(nn.Linear(self.hparams.model_dim, 64), nn.Softmax(dim=2))
        self.time_num_head = nn.Sequential(nn.Linear(self.hparams.model_dim, 32), nn.Softmax(dim=2))
        self.time_denom_head = nn.Sequential(nn.Linear(self.hparams.model_dim, 32), nn.Softmax(dim=2))
        self.bar_head = nn.Sequential(nn.Linear(self.hparams.model_dim, 1024), nn.Softmax(dim=2))
        self.inst_head = nn.Sequential(nn.Linear(self.hparams.model_dim, 128), nn.Softmax(dim=2))
        self.tmp_head = nn.Sequential(nn.Linear(self.hparams.model_dim, 32), nn.Softmax(dim=2))
        self.vel_head = nn.Sequential(nn.Linear(self.hparams.model_dim, 32), nn.Softmax(dim=2))
        
        #bert is encoder only
        xformer_config = [
        {
            "reversible": False,  # Turn on to test the effect of using reversible layers
            "block_type": "encoder",
            "num_layers": self.hparams.num_layers,
            "dim_model": self.hparams.model_dim,
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
                "hidden_layer_multiplier": self.hparams.hidden_layer_multiplier, 
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
        if batch is None:
            return torch.tensor(0.0) #return 0 loss if batch is empty
        
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
        
        assert x.shape[0]*x.shape[1]*x.shape[2] != 0, "x is empty"
        
        #where x is all zero
        pad_mask = (x.sum(dim=2) == 0).long()
        
        x = self.mixer(x)
        
        #[0,1] mask for each token in sequence
        mask = (torch.rand((x.shape[0], x.shape[1]), device=self.device) < 0.8).long() #mwhen random number is less than 0.8, keep the note
        #Mask out notes according to mask
        x = x*mask.unsqueeze(-1)
        #Add embedding of MASK token where mask is 0
        MASK = self.special_tokens(torch.zeros((x.shape[0], x.shape[1]), device=self.device).long())*((1-mask).unsqueeze(-1))
        x = x + MASK
        
        x = x*(1-pad_mask.unsqueeze(-1)) #set pad tokens to 0
        x = x + self.special_tokens(torch.ones((x.shape[0], x.shape[1]), device=self.device).long())*((1-pad_mask).unsqueeze(-1)) #add embedding of pad token where pad_mask is 1
        
        
        
        y = self.transformer(x)

        
        pitch_loss = F.cross_entropy(self.pitch_head(y).permute(0, 2, 1), pitch.long())
        dur_loss = F.cross_entropy(self.dur_head(y).permute(0, 2, 1), dur.long())
        offset_loss = F.cross_entropy(self.offset_head(y).permute(0, 2, 1), off.long())
        tden_loss = F.cross_entropy(self.time_denom_head(y).permute(0, 2, 1), tden.long())
        tnum_loss = F.cross_entropy(self.time_num_head(y).permute(0, 2, 1), tnum.long())
        bar_loss = F.cross_entropy(self.bar_head(y).permute(0, 2, 1), bar.long())
        inst_loss = F.cross_entropy(self.inst_head(y).permute(0, 2, 1), inst.long())
        tmp_loss = F.cross_entropy(self.tmp_head(y).permute(0, 2, 1), tmp.long())
        vel_loss = F.cross_entropy(self.vel_head(y).permute(0, 2, 1), vel.long())
        
        loss = pitch_loss + dur_loss + offset_loss + tden_loss + tnum_loss + bar_loss + inst_loss + tmp_loss + vel_loss
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())