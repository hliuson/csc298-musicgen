import argparse
import os
import sys
import time

import muspy
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from autoencoder import *
from data import getdatasets
from model import *


### Split the pianorolls into 512-long segments and return a stacked tensor of shape (N, 512, 128),
def get_cuts(pianorolls):
    cuts = []
    for pianoroll in pianorolls:
        for i in range(0, pianoroll.shape[0] - 512, 512):
            cuts.append(torch.Tensor(pianoroll[i : i + 512, :]))
            
    max_cuts = 32
    cuts = torch.stack(cuts)
    if len(cuts) > max_cuts:
        cuts = cuts[torch.randperm(cuts.shape[0])][:max_cuts]
    return cuts

#return L-long segments of the pianorolls for the autoencoder with random start points,
# and a tensor of shape (N, L, 128)
class MiniCuts():
    def __init__(self, cut_length = 32):
        self.L = cut_length
    
    def __call__(self, pianorolls):
        cuts = []
        for pianoroll in pianorolls:
            for i in range(0, pianoroll.shape[0] - self.L, self.L):
                cuts.append(torch.Tensor(pianoroll[i : i + self.L, :]))
        stack = torch.stack(cuts)
        return stack

def main(*args, **kwargs):
    #argparse options for new and continue training
    parser = argparse.ArgumentParser()
    parser.add_argument('--loadFrom', type=str, default=None)
    parser.add_argument('--saveTo', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--wandbcomment', type=str, default=None)
    parser.add_argument('--conv_dim', type=int, default=None)
    parser.add_argument('--kernel_size', type=int, default=None)
    parser.set_defaults(new=True)
    args = parser.parse_args(args)
    
    #silence wandb
    os.environ['WANDB_SILENT'] = "true"
        
    train, test = download_dataset()
    
    train_autoencoder(args, train, test)

def download_dataset():
    #Load MAESTRO dataset into folder structure
    _ = muspy.datasets.MAESTRODatasetV3(root="data", download_and_extract=True)
    return getdatasets()

def train_autoencoder(args, train, test):
    if args.batch_size is None:
        args.batch_size = 1
    
    if args.workers is None:
        args.workers = 1
    
    if args.wandbcomment is None:
        args.wandbcomment = "autoencoder"
    
    minicuts = MiniCuts(cut_length=32)
    
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=minicuts)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=minicuts)

    model = ConvAutoencoder()
    
    wandblogger = pl.loggers.WandbLogger(project="test-project")
    wandblogger.watch(model, log="all")
    #wandb.config["comment"] = args.wandbcomment
    
    ddp = pl.strategies.DDPStrategy(process_group_backend="nccl", find_unused_parameters=False)
    
    trainer = pl.Trainer(default_root_dir=args.saveTo, accelerator="gpu",
                         devices=torch.cuda.device_count(), max_epochs=args.epochs, logger=wandblogger,strategy=ddp,
                         callbacks=[pl.callbacks.ModelCheckpoint(dirpath=args.saveTo, monitor="val_loss", mode="min", save_top_k=1, save_last=True, verbose=True),])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader, ckpt_path=args.loadFrom)
    
if __name__ == '__main__':
    # Run the main function with the arguments passed to the script
    main(*sys.argv[1:])
