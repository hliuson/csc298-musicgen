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
from data import *
from model import *
from sequence import *
from midiseq import *


def main(*args, **kwargs):
    #argparse options for new and continue training
    parser = argparse.ArgumentParser()
    parser.add_argument('--loadFrom', type=str, default=None)
    parser.add_argument('--saveTo', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.set_defaults(new=True)
    args = parser.parse_args(args)
    
    #silence wandb
    os.environ['WANDB_SILENT'] = "true"
    
    if args.saveTo is None:
        print("Please specify a save location")
        return
        
    train, test = midi_dataset()
    
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=BERTTokenBatcher())
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=BERTTokenBatcher())
    
    model = MidiFormer()
    
    wandblogger = pl.loggers.WandbLogger(project="midi-bert")
    wandblogger.watch(model, log="all")
    
    ddp = pl.strategies.DDPStrategy(process_group_backend="nccl", find_unused_parameters=False)
    
    trainer = pl.Trainer(default_root_dir=args.saveTo,  amp_level="O2", amp_backend="apex", accelerator="gpu",
                         devices=torch.cuda.device_count(), max_epochs=args.epochs, logger=wandblogger, strategy=ddp,
                         callbacks=[pl.callbacks.ModelCheckpoint(dirpath=args.saveTo, monitor="val_loss", mode="min", save_top_k=1, save_last=True, verbose=True),])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader, ckpt_path=args.loadFrom)

if __name__ == '__main__':
    main(*sys.argv[1:])