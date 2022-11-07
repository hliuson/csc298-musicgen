from model import *
import muspy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import MidiDataset
import wandb
import time
import os

### Split the pianorolls into 512-long segments and return a stacked tensor of shape (N, 512, 128),
def get_cuts(pianorolls):
    cuts = []
    for pianoroll in pianorolls:
        for i in range(0, pianoroll.shape[0] - 512, 512):
            cuts.append(torch.Tensor(pianoroll[i : i + 512, :]))
            
    max_cuts = 256
    cuts = torch.stack(cuts)
    if len(cuts) > max_cuts:
        cuts = cuts[torch.randperm(cuts.shape[0])][:max_cuts]
    return cuts
    

def main():
    global data 
    global dataset
    #silence wandb
    os.environ['WANDB_SILENT'] = "true"
    
    #Load MAESTRO dataset into folder structure
    _ = muspy.datasets.MAESTRODatasetV3(root="data", download_and_extract=True)
    
    #Define our own dataset class which is pickleable
    dataset = MidiDataset("data/maestro-v3.0.0")
    
    ### Train an LSTM model on the MAESTRO dataset to predict the next 32 steps of a pianoroll.

    model = get_baseline_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    batch = 1

    train_loader = DataLoader(dataset, batch_size=batch, shuffle=False, collate_fn=get_cuts, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch, shuffle=False, collate_fn=get_cuts, num_workers=4, pin_memory=True)
    
    
    epoch = 0

    path = "./checkpoints"
    for file in os.listdir(path):
        if file.endswith(".pt"):  
            if file != "checkpoint.pt":
                continue          
            path = path + "/"+ file
            print("Found incomplete training session, loading model")
            break
    
    #Load model, optimizer, epoch, run
    if path != "./checkpoints":
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        run = checkpoint['run']
        print("Loaded model from previous training session")
    else :
        run = None
    
    
        
    if path == "checkpoints/":
        print("No previous training session found")
        # Initialize wandb
        wandb.init(project="test-project", entity="csc298-hliuson-dchien")
        wandb.config = {
            "model": "LSTM",
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "batch_size": batch,
        }
    else:
        wandb.init(project="test-project", entity="csc298-hliuson-dchien", id=run)
        
    wandb.watch(model, log="all")
    
    start = time.time()

    ### Core training loop
    while epoch < 10:
        # Train
        model.train()
        train_loss = 0
        for n, pianorolls in enumerate(train_loader):
            if n % 100 == 0:
                print("Epoch: " + str(epoch) + " Batch: " + str(n))
            pianorolls = pianorolls.to(device)
            
            outputs= model(pianorolls[:, :-32, :])
            # Get the rest of the 32 steps
            
            
            # Compute loss against the last 32 steps
            loss = criterion(
                outputs,
                pianorolls[:, -32:, :],
            )
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * pianorolls.shape[0]
            wandb.log({"train_loss": loss.item()})
        train_loss /= len(train_loader.dataset)
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for pianorolls in val_loader:
                pianorolls = pianorolls.to(device)
                outputs= model(pianorolls[:, :-32, :])
                # Get the rest of the 32 steps
                
                
                # Compute loss against the last 32 steps
                loss = criterion(
                    outputs,
                    pianorolls[:, -32:, :],
                )
                wandb.log({"val_loss": loss.item()})
                val_loss += loss.item() * pianorolls.shape[0]
        val_loss /= len(val_loader.dataset)
        # Print results
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss = {train_loss:.4f}, "
            f"val_loss = {val_loss:.4f}"
        )
        #log time elapsed for epoch
        end = time.time()
        wandb.log({"time": end - start})
        
        #create path
        path = "checkpoints/" + "checkpoint" + ".pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'run': wandb.run.id,
            }, path)

        start = time.time()
        epoch += 1

if __name__ == '__main__':
    main()
    