#In this file, we are going to predict the next measure in a piece based on our trained models
#To do so, we will feed a .midi file with an unfilled last measure to the model
#We will encode this midi file, and then use our RNN to predict the next tokens
#We will then decode these tokens into a midi file, and save it to the output folder

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
from lstm import *
from gru import *
#from sequence import *
from model import *
from infer import *
import pypianoroll

def main(*args, **kwargs):
    #argparse options
    parser = argparse.ArgumentParser()
    parser.add_argument('--loadFrom', type=str, default=None)
    parser.add_argument('--saveTo', type=str, default=None)
    #Input midi file
    parser.add_argument('--input', type=str, default=None)
    parser.set_defaults(new=True)
    args = parser.parse_args(args)

    #If --input is None, raise error saying "We cannot predict on nothing!"
    if args.input is None:
        raise ValueError("We cannot predict on nothing!")
    
    #If --saveTo is None, default save to predicted
    if args.saveTo is None:
        args.saveTo = "predicted"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load the models
    autoencoder = load_autoencoder()
    gru_name = "gru-test"
    gru = load_gru(gru_name)

    #Load the input midi file
    input_midi = muspy.outputs.pianoroll.to_pianoroll_representation(muspy.read(args.input))
    original = input_midi
    input_midi = input_midi[None,:]
    input_midi = EncoderDataset(dataset=input_midi,embedder=autoencoder)
    #Load into dataloader
    dataloader = DataLoader(input_midi, batch_size=1, shuffle=False)
    #Get shapes of data in dataloader
    trainer = pl.Trainer(accelerator='cpu')
    #Predict
    predictions = trainer.predict(gru, dataloader)
    #Decode
    predictions = torch.tensor(predictions[0]).squeeze()
    decoded = autoencoder.decode(predictions)
    decoded = decoded.to(device).reshape(-1, 128)
    print(original.shape, decoded.shape)
    #Turn back to midi output
    #Save to output folder
    activation = decoded > 0.5
    activation = activation.numpy().astype(int)
    tempo = 240*np.ones((activation.shape[0], 1))
    roll = pypianoroll.Multitrack(tracks=[pypianoroll.BinaryTrack(pianoroll=activation)], tempo=tempo)
    pypianoroll.write(os.path.join(args.saveTo, args.input), roll)
    #Print Done!
    print("Done!")

def load_simpleautoencoder(model_name):
    #load the last checkpoint of the model using pytorch-lightning
    file = os.path.join("checkpoints", model_name, "last.ckpt")
    model = SimpleAutoencoder.load_from_checkpoint(file, conv_dim=4, kernel=13)
    model.eval()
    return model

def load_gru(model_name):
    #load the last checkpoint of the model using pytorch-lightning
    file = os.path.join("checkpoints", model_name, "last.ckpt")
    model = autoGRU.load_from_checkpoint(file, input_size=128, hidden_dim=256, dropout=0.3)
    model.eval()
    return model

if __name__ == '__main__':
    # Run the main function with the arguments passed to the script
    main(*sys.argv[1:])