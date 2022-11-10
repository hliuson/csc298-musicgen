import torch
import os
import muspy
import numpy as np
from data import MidiDataset
from train import get_cuts
from torch.utils.data import DataLoader
from model import get_baseline_model

def main():
    activation_threshold = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load("checkpoints/checkpoint.pt")
    model = get_baseline_model().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    #empty /midi folder
    for root, dirs, files in os.walk("midi"):
        for file in files:
            os.remove(os.path.join(root, file))
    
    dataset = MidiDataset("data/maestro-v3.0.0")
    val_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=get_cuts, num_workers=4, pin_memory=True)
    import time
    start = time.time()
    for i, data in enumerate(val_loader):
        if i == 5:
            break
        data = data.to(device)
        output = model(data)
        output = output.detach().cpu().numpy()
        output = np.where(output > activation_threshold, 1, 0) # activation threshold
        
        
        #attach output to data to get the full pianoroll
        output = np.concatenate((data.detach().cpu().numpy(), output), axis=1).astype(int)
        #unstack the output into a list of numpy arrays
        output = np.split(output, output.shape[0], axis=0)
        output = np.squeeze(output)
        
        for j in range(len(output)):
            #randomly choose a few outputs to save as midi files
            if np.random.randint(0, 100) < 99:
                continue
            #use muspy to convert pianoroll to midi file
            music = muspy.inputs.pianoroll.from_pianoroll_representation(output[j])
            music.print()
            music.write_midi("midi/output{}_{}.midi".format(i, j))
    end = time.time()
    print(end - start)
    
    
    
if __name__ == '__main__':
    main()