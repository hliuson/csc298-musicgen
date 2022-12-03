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


class SequenceEmbedder():
    def __init__(self, autoencoder, cut_length = 32, embedding_dim = 128):
        self.L = cut_length
        self.autoencoder = autoencoder
    
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
    parser.set_defaults(new=True)
    args = parser.parse_args(args)
    
    #silence wandb
    os.environ['WANDB_SILENT'] = "true"
        
    train, test = download_dataset()
    
    model_name = "autoencoder-simple-4-13"
    #4-13 achieves 93$ IOU accuracy on validation set, and reduces dimension fairly aggressively,
    #reducing 32x128 to 128. 
    autoencoder = load_simpleautoencoder(model_name)
    embed = SequenceEmbedder(autoencoder)
    
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    #print the shape of the first batch
    for batch in train_loader:
        print(batch.shape)
        embeddings = embed(batch)
        print(embeddings.shape)
        break
    
def load_simpleautoencoder(model_name):
    #load the last checkpoint of the model using pytorch-lightning
    file = os.path.join("checkpoints", model_name, "last.ckpt")
    model = SimpleAutoencoder.load_from_checkpoint(file, conv_dim=4, kernel=13)
    model.eval()
    return model

if __name__ == '__main__':
    main(*sys.argv[1:])