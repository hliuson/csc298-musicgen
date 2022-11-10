import torch 
import os
import muspy
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
    
def getdatasets(split = 0.9):
    print("Loading dataset...")
    root = "./data/maestro-v3.0.0"
    paths = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".midi"):
                paths.append(os.path.join(root, file))
                
    print("Splitting dataset...")
    trainidx = np.random.choice(len(paths), int(len(paths) * split), replace=False)
    testidx = np.setdiff1d(np.arange(len(paths)), trainidx)
    
    print("Creating datasets...")
    train = MidiDataset(paths, trainidx)
    test = MidiDataset(paths, testidx)
    
    return train, test
    
    
    