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
    
    #Load the models
    autoencoder_name = "autoencoder-simple-4-13"
    autoencoder = load_simpleautoencoder(autoencoder_name)
    gru_name = "gru-test"
    gru = load_gru(gru_name)

    #Load the input midi file
    input_midi = muspy.outputs.pianoroll.to_pianoroll_representation(muspy.read(args.input))
    input_midi = input_midi[None,:]
    input_midi = EncoderDataset(dataset=input_midi,embedder=autoencoder)
    #Load into dataloader
    dataloader = DataLoader(input_midi, batch_size=1, shuffle=False)
    print(len(dataloader))
    #Trainerus=1)
    trainer = pl.Trainer(accelerator='cpu')
    #Predict
    predictions = trainer.predict(gru, dataloader)
    #Decode
    #Convert predictions[0] to ndarray
    predictions = np.array(predictions[0])
    #decoded = autoencoder.decode(predictions[0])
    #Turn back to midi output
    #Save to output folder
    muspy.from_pianoroll_representation(predictions).write(os.path.join(args.saveTo, args.input))
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