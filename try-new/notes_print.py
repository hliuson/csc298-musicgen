import os
import numpy as np
from music21 import *
import torch
import torch.nn as nn

NOTE_SIZE = 128

def midi2tensor(midi):
    stream = converter.parse(midi)
    notes = stream.flat.getElementsByClass('Note')
    pitches = []
    durations = []
    pos = []
    for i in range(len(notes)):
        pitches.append(notes[i].pitch)
        durations.append(notes[i].duration.quarterLength*8)
        pos.append(notes[i].offset)
    pitches = [pitch.midi for pitch in pitches]
    notes = np.array([pitches, durations, pos]).T
    notes = notes[None,:,:].astype(int)
    return notes

def getdatasets():
    root = "/home/dchien/csc298_musicgen-main/data/maestro-v3.0.0"
    paths = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".midi"):
                paths.append(os.path.join(root, file))
    trainidx = np.random.choice(len(paths), int(len(paths) * 0.9), replace=False)
    testidx = np.setdiff1d(np.arange(len(paths)), trainidx)
    train = [midi2tensor(paths[i]) for i in trainidx]
    test = [midi2tensor(paths[i]) for i in testidx]
    return train, test

print(getdatasets())

