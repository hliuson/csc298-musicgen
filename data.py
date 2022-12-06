import torch 
import os
import muspy
import music21
import numpy as np

#The muspy implementation of .to_pytorch_dataset() returns a class which is not pickleable,
# so we define our own class which is pickleable.

### Define a dataset class in torch.utils.data.Dataset format.
### Data is stored as .midi files in a folder structure.
### The dataset class will need to convert the .midi files to pianoroll format using muspy.

class MidiDataset(torch.utils.data.Dataset):
    def __init__(self, files, indices):
        self.files = files
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        return muspy.outputs.pianoroll.to_pianoroll_representation(muspy.read(self.files[self.indices[index]]))
  
class MidiTokenDataset(torch.utils.data.Dataset):
    def __init__(self, files, indices):
        self.files = files
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        #use music21 to tokenize the midi file
        midi = music21.converter.parse(self.files[self.indices[index]])
        notes = midi.flat.getElementsByClass('Note')
        
        #128 pitches
        pitches = np.zeros((len(notes)), dtype=np.float32)
        
        #durations are quantized into bins with 1/32nd note resolution
        #A quarter note has duration 1 in midi, under our quantization scheme, a quarter note has duration 8
        durations = np.zeros((len(notes)), dtype=np.int32)
        
        #
        positions = np.zeros((len(notes)), dtype=np.int32)
        
        for i, note in enumerate(notes):
            pitches[i] = note.pitch.midi
            if note.duration.quarterLength > 0:
                durations[i] = 128
            else:
                durations[i] = int(note.duration.quarterLength * 8)
            positions[i] = int(note.offset * 8) #quantize to 1/32nd note resolution
        return torch.tensor([pitches, durations, positions], dtype=torch.float32).T #shape (num_notes, 3)

        
    
def getdatasets(split = 0.9, embedder = None, L=32, embed_length = 128):
    root = "./data/maestro-v3.0.0"
    paths = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".midi"):
                paths.append(os.path.join(root, file))

    trainidx = np.random.choice(len(paths), int(len(paths) * split), replace=False)
    testidx = np.setdiff1d(np.arange(len(paths)), trainidx)
    
    train = MidiDataset(paths, trainidx)
    test = MidiDataset(paths, testidx)
    
    if embedder is not None:
        train = EncoderDataset(train, embedder)
        test = EncoderDataset(test, embedder)
    
    return train, test

def midi_dataset(split=0.95):
    root = "./data/maestro-v3.0.0"
    paths = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".midi"):
                paths.append(os.path.join(root, file))

    trainidx = np.random.choice(len(paths), int(len(paths) * split), replace=False)
    testidx = np.setdiff1d(np.arange(len(paths)), trainidx)
    
    train = MidiTokenDataset(paths, trainidx)
    test = MidiTokenDataset(paths, testidx)
    
    return train, test

def download_lakh():
    #Load MAESTRO dataset into folder structure
    #This file is in: /home/username/....
    #We want: /scratch/username/....
    
    
    datapath = "/home/hliuson/data"
    _ = muspy.datasets.LakhMIDIDataset(root=datapath, download_and_extract=True)
    return getdatasets()

#Use the sequence embedder to preprocess the data and save it to disk
#The sequence embedder takes a sequence of length N and returns a sequence of length N//L.
class EncoderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, embedder, L=32, embed_dim=128, MAX_LEN=256):
        self.dataset = dataset
        self.embedder = embedder.to("cpu")
        self.L = L
        self.embed_dim = embed_dim
        
        self.MAX_LEN = MAX_LEN

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        #partition the sequence into L-length chunks
        seq = self.dataset[index]
        L = self.L
        N = seq.shape[0]

        #pad the sequence with zeros if it is not a multiple of L
        seq = np.pad(seq, ((0, L - (N % L)), (0, 0)), mode="constant")
        
        #reshape the sequence into a 3D tensor of shape (num_chunks, L, 128)
        seq = seq.reshape((-1, L, self.embed_dim))
        
        #cast the sequence to a torch tensor
        seq = torch.tensor(seq, dtype=torch.float32)
        
        #embed the sequence
        with torch.no_grad():
            seq = self.embedder.encode(seq)
        
        #reshape the sequence into a 2D tensor of shape (num_chunks * L, 128)
        seq = seq.reshape((-1, self.embed_dim))
        
        #choose random subsequence of length MAX_LEN if the sequence is longer than MAX_LEN
        if seq.shape[0] > self.MAX_LEN:
            start = np.random.randint(0, seq.shape[0] - self.MAX_LEN)
            seq = seq[start:start+self.MAX_LEN]
        
        return seq
        
    
def main():
    train, test = download_lakh()
    print(len(train), len(test))
    print(train[0].shape)
    print(test[0].shape)

if __name__ == "__main__":
    main()