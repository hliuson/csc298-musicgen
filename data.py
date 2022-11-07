import torch 
import os
import muspy

#The muspy implementation of .to_pytorch_dataset() returns a class which is not pickleable,
# so we define our own class which is pickleable.

### Define a dataset class in torch.utils.data.Dataset format.
### Data is stored as .midi files in a folder structure.
### The dataset class will need to convert the .midi files to pianoroll format using muspy.

class MidiDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.files = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith(".midi"):
                    self.files.append(os.path.join(root, file))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        return muspy.outputs.pianoroll.to_pianoroll_representation(muspy.read(self.files[index]))