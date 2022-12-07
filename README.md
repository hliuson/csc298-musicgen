# Phrase Encodings for Music Synthesis
Harry Liuson
Derek Chien

# Contributing
Using: 
Python 3.10

# Data
Raw data is .MIDI files, converted to pianorolls via MusPy. Pianorolls are Nx128 boolean tensors where each 128-dimensional vector represents which
notes are active or inactive at that time step.

The autoencoder converts this to (N // embed_length) x (embedding_dim).

(embed_length x 128) -> (1 x embedding_dim)