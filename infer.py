import torch
import os
import muspy
import numpy as np
import pypianoroll
from data import *
from train_sequence import *
from evaluate import *
from train_autoencoder import *
from torch.utils.data import DataLoader
from autoencoder import *

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #autoencoder = load_simpleautoencoder("autoencoder-simple-4-13").to(device)
    #sequence = TransformerSequence.load_from_checkpoint("checkpoints/sequence-transformer-test/last.ckpt").to(device)
    autoencoder = load_autoencoder().to(device)
    train, test = getdatasets()
    test_loader = DataLoader(test, batch_size=1, shuffle=True, num_workers=1)
    minibatch_loader = DataLoader(test, batch_size=1, shuffle=True, num_workers=1, collate_fn=MiniCuts())

    #Feed 64 embedded tokens into the sequence model and autoregressively generate the next 64 tokens.
    #Decode the 128 tokens into piano roll and save to midi.
    #empty the generated folder
    os.system("rm -rf ./generated")
    os.makedirs("generated", exist_ok=True)
    
    no_examples = 0
    iou = {
        "0.5": 0,
        "0.6": 0,
        "0.7": 0,
        "0.8": 0,
        "0.9": 0,
    }
    with torch.no_grad():
        for batch_idx, data in enumerate(minibatch_loader):
            data = data.to(device).reshape(-1, 32, 128)
            reconstruction = autoencoder(data).reshape(1, -1, 128)
            data = data.reshape(1, -1, 128)
            iou = iou_score(reconstruction, data, 0.5)
            print("IoU for sample {}: {}".format(batch_idx, iou))
            write_midi("generated/reconstruction_{}.mid".format(batch_idx), reconstruction, 0.5)
            write_midi("generated/original_{}.mid".format(batch_idx), data, 0.5)
    #for i, (embeddings) in enumerate(test_loader):
    #    embeddings = embeddings.to(device).reshape(1, -1, 128)
    #    embeddings = embeddings[:, :64, :]
    #    reconstructed = autoencoder.decode(embeddings.reshape(-1, 128)).reshape(-1, 128)
    #    write_midi("generated/reconstructed_{}.mid".format(i), reconstructed, 0.5)
        
        
    
    '''with torch.no_grad():
        for i, (embeddings) in enumerate(test_loader):
            embeddings = embeddings.to(device).reshape(1, -1, 128)
            embeddings = embeddings[:, :64, :]
            reconstructed = autoencoder.decode(embeddings.reshape(-1, 128)).reshape(-1, 128)
            output = sequence(embeddings).reshape(1, -1, 128)
            #now infer the next 64 tokens, cat each one to the output.
            #Sequence inputs and outputs shape is (batch_size, seq_len, 128)
            for j in range(64):
                output = torch.cat((output, sequence(output[:, -1, :]).reshape(1, -1, 128)), dim=1)
            output = output.reshape(-1, 128)
            output = autoencoder.decode(output).reshape(-1, 128)
            #save to midi
            write_midi("generated/example_{}.mid".format(i), output, 0.5)
            write_midi("generated/reconstructed_{}.mid".format(i), reconstructed, 0.5)

            if i == 20:
                return
    '''

def write_midi(filename, data, threshold):
    data = data.cpu().reshape(-1, 128)
    #average-pool the data in time to suppress noise
    pool = torch.nn.AvgPool1d(5, stride=1, padding=2)
    #data = pool(data)
    
    #data is a (N, 128) tensor
    #write to midi file using muspy
    activation = data > threshold
    
    # integer numpy array 
    activation = activation.numpy().astype(int)
    tempo = 240*np.ones((activation.shape[0], 1))
    roll = pypianoroll.Multitrack(tracks=[pypianoroll.BinaryTrack(pianoroll=activation)], tempo=tempo)
    # make tempo pretty high
    pypianoroll.write(filename, roll) 
    
if __name__ == '__main__':
    main()
