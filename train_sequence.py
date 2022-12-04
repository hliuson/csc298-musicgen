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
from lstm import *
from model import *


class SequenceEmbedder():
    def __init__(self, autoencoder, cut_length = 32, embedding_dim = 128):
        self.L = cut_length
        self.autoencoder = autoencoder
        self.embedding_dim = embedding_dim
    
    #for each pianoroll with length L_i, slice it into segments of length L.
    #Let the longest pianoroll be L_max.
    #Run the model on each segment and return a tensor of shape (N, L_max, embedding_dim)
    def __call__(self, pianorolls):
        
        #Transform the pianorolls into a tensor of shape (N, L_max // L, 128)
        #Then run the encoder on each segment and return a tensor of shape (N, L_max // L, embedding_dim)
        def transform(pianoroll):
            cuts = []
            for i in range(0, pianoroll.shape[0] - self.L, self.L):
                cuts.append(torch.Tensor(pianoroll[i : i + self.L, :]))
            stack = torch.stack(cuts)
            return self.autoencoder.encode(stack)
        
        embeddings = []
        for pianoroll in pianorolls:
            print(f"{len(embeddings)/len(pianorolls)*100}% done")
            embeddings.append(transform(pianoroll)) 
        
        #Pad the embeddings with zeros so that each embedding is of shape (L_max // L, embedding_dim)
        max_length = max([len(embedding) for embedding in embeddings])
        for i, embedding in enumerate(embeddings):
            if len(embedding) < max_length:
                embeddings[i] = torch.cat([embedding, torch.zeros(max_length - len(embedding), self.embedding_dim)])
        #Note: we may want to consider using self.encoder(zeros) instead of zeros.
        #But this works for now
        
        return torch.stack(embeddings)

def main(*args, **kwargs):
    #argparse options for new and continue training
    parser = argparse.ArgumentParser()
    parser.add_argument('--loadFrom', type=str, default=None)
    parser.add_argument('--saveTo', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--wandbcomment', type=str, default=None)
    parser.set_defaults(new=True)
    args = parser.parse_args(args)
    
    if args.batch_size is None:
        args.batch_size = 1
    
    if args.workers is None:
        args.workers = 1
    
    if args.wandbcomment is None:
        args.wandbcomment = "sequence"

    #silence wandb
    os.environ['WANDB_SILENT'] = "true"
    train, test = download_dataset()
    print(train.type())
    #Sample 10% of train. train is a data.MidiDataset object
    train_sample = torch.utils.data.Subset(train, np.random.choice(len(train), int(len(train) * 0.01), replace=False))
    test_sample = torch.utils.data.Subset(test, np.random.choice(len(test), int(len(test) * 0.01), replace=False))
    model_name = "autoencoder-simple-4-13"    #4-13 achieves 93$ IOU accuracy on validation set, and reduces dimension fairly aggressively,
    #reducing 32x128 to 128. 
    autoencoder = load_simpleautoencoder(model_name)
    embed = SequenceEmbedder(autoencoder)
    train_sample = embed(train_sample)
    test_sample = embed(test_sample)

    #Make sure the data is in FloatTensor format. Print its type
    print(train_sample.type())
    print(test_sample.type())
    print(train_sample[:,:-1,:].shape)
    print(train_sample[:,1:,:].type())
    print(test_sample.shape)

    train_loader = DataLoader(train_sample, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_sample, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    #Initialize model = autoLSTM(input_size,hidden_dim,dropout=0.2). Get input_size right
    #As a recap, the input to autoLSTM is of shape (batch_size, seq_len, embedding_dim), where embedding_dim = 128
    #The output is of shape (batch_size, seq_len, embedding_dim) where embedding_dim = 128
    #Please initialize the model with the right input shape to input_size!
    model = autoLSTM(128, 128, dropout=0.2)



    wandblogger = pl.loggers.WandbLogger(project="test-project")
    wandblogger.watch(model, log="all")
    
    ddp = pl.strategies.DDPStrategy(process_group_backend="nccl", find_unused_parameters=False)
    
    trainer = pl.Trainer(default_root_dir=args.saveTo, accelerator="gpu",
                         devices=torch.cuda.device_count(), max_epochs=args.epochs, logger=wandblogger,strategy=ddp,
                         callbacks=[pl.callbacks.ModelCheckpoint(dirpath=args.saveTo, monitor="val_loss", mode="min", save_top_k=1, save_last=True, verbose=True),])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
        
def load_simpleautoencoder(model_name):
    #load the last checkpoint of the model using pytorch-lightning
    file = os.path.join("checkpoints", model_name, "last.ckpt")
    model = SimpleAutoencoder.load_from_checkpoint(file, conv_dim=4, kernel=13)
    model.eval()
    return model

if __name__ == '__main__':
    main(*sys.argv[1:])