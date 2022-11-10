from model import *
from autoencoder import *
import muspy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import getdatasets
import wandb
import time
import os
import argparse
import sys
import numpy as np

### Split the pianorolls into 512-long segments and return a stacked tensor of shape (N, 512, 128),
def get_cuts(pianorolls):
    cuts = []
    for pianoroll in pianorolls:
        for i in range(0, pianoroll.shape[0] - 512, 512):
            cuts.append(torch.Tensor(pianoroll[i : i + 512, :]))
            
    max_cuts = 32
    cuts = torch.stack(cuts)
    if len(cuts) > max_cuts:
        cuts = cuts[torch.randperm(cuts.shape[0])][:max_cuts]
    return cuts

#return 32-long segments of the pianorolls for the autoencoder with random start points,
# and a tensor of shape (N, 32, 128)
def get_mini_cuts(pianorolls):
    cuts = []
    for _ in range(32):
        pianoroll = pianorolls[np.random.randint(0, len(pianorolls))]
        start = np.random.randint(0, pianoroll.shape[0] - 32)
        cuts.append(torch.Tensor(pianoroll[start : start + 32, :]))
    return torch.stack(cuts)

def main(*args, **kwargs):
    #argparse options for new and continue training
    parser = argparse.ArgumentParser()
    parser.add_argument('--continueFrom', type=str, default=None)
    parser.add_argument('--saveTo', type=str, default=None)
    parser.add_argument('--autoencoder', dest='autoencoder', action='store_true')
    parser.add_argument('--LSTM', dest='autoencoder', action='store_false')
    parser.add_argument('--epochs', type=int, default=10)
    parser.set_defaults(new=True)
    args = parser.parse_args(args)
    
    #silence wandb
    os.environ['WANDB_SILENT'] = "true"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(torch.cuda.device_count()):
        print("GPU ", i, ": ", torch.cuda.get_device_name(i))
    
    cont = args.continueFrom is not None
    if cont:
        if not os.path.exists(args.continueFrom):
            print("Checkpoint file does not exist, starting new run")
            cont = False
    
    print("Using device: ", device)
    if cont:
        checkpoint = torch.load(args.continueFrom)
        # verify that the model is the same as the one we are trying to continue training
        if args.autoencoder:
            assert isinstance(checkpoint['model'], ConvAutoEncoder)
        if not args.autoencoder:
            assert isinstance(checkpoint['model'], LSTMModel)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        epoch = checkpoint['epoch']
        run = checkpoint['run']
        wandb.init(project="test-project", resume="allow", id=run)
        
    else:
        if args.autoencoder:
            model = ConvAutoEncoder().to(device)
        else: 
            model = LSTMModel.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epoch = 0
        
    if args.saveTo is None:
        print("Must specify a save location")
        return
        
    train, test = download_dataset()
    
    if args.autoencoder:
        train_autoencoder(args, model=model, optimizer=optimizer,
                          epoch=epoch, device=device, train=train, test=test)
    else:
        train_LSTM(args)

def download_dataset():
    #Load MAESTRO dataset into folder structure
    _ = muspy.datasets.MAESTRODatasetV3(root="data", download_and_extract=True)
    return getdatasets()

def train_autoencoder(args, model=None, optimizer=None, epoch=0, train_loader = None, val_loader = None,
                      device = None, train = None, test = None):
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, collate_fn=get_mini_cuts)
    
    val_loader = DataLoader(test, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, collate_fn=get_mini_cuts)
    print("Initializing wandb")
    
    # Initialize wandb
    wandb.init(project="test-project", entity="csc298-hliuson-dchien")
    wandb.config = {
        "learning_rate": 1e-3,
        "batch_size": 1,
        "num_epochs": 10,
        "model": "ConvAutoEncoder",
    }
    for epoch in range(epoch, args.epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / len(train_loader)))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                data = data.to(device)
                output = model(data)
                val_loss += criterion(output, data).item()

        wandb.log({"epoch": epoch,
                   "loss": train_loss / len(train_loader),
                   "val_loss": val_loss / len(val_loader)})
        torch.save(
            {
                "model": model,
                "optimizer": optimizer,
                "epoch": epoch,
                "run": wandb.run.id,
            },
            args.saveTo,
        )
    
def train_LSTM(args, model=None, optimizer=None, epoch=0, train_loader = None, val_loader = None, device = None, dataset = None):

    criterion = nn.CrossEntropyLoss()

    batch = 1
    threshold = 0.5

    train_loader = DataLoader(dataset["train"], batch_size=batch, shuffle=False, collate_fn=get_cuts, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset["test"], batch_size=batch, shuffle=False, collate_fn=get_cuts, num_workers=4, pin_memory=True)   
        
    wandb.watch(model, log="all")
    
    start = time.time()

    ### Core training loop
    while epoch < 10:
        # Train
        model.train()
        train_loss = 0
        train_accuracy = 0
        train_samples = 0
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

            train_loss += loss.item()  * pianorolls.shape[0]
            #this is an instance of multi-label classification, so we need some threshold to determine if a note is on or off
            activation = outputs > threshold
            pianorolls = pianorolls.bool()
            #Define accuracy as intersection over union of the predicted and actual notes per timestep
            intersection = (activation & pianorolls[:, -32:, :]).sum(dim=2)
            union = (activation | pianorolls[:, -32:, :]).sum(dim=2)
            # if union is 0, then the intersection is also 0, so we can just set the union to 1
            if (union == 0).any():
                union[union == 0] = 1
            # train_accuracy += (intersection / union).sum(dim=1).mean().item() * pianorolls.shape[0]
            train_accuracy += (intersection / union).mean().item() * pianorolls.shape[0]
            train_samples += pianorolls.shape[0]
        train_loss /= train_samples
        train_accuracy /= train_samples
        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_samples = 0
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
                #this is an instance of multi-label classification, so we need some threshold to determine if a note is on or off
                activation = outputs > threshold
                pianorolls = pianorolls.bool()
                #Define accuracy as intersection over union of the predicted and actual notes per timestep
                intersection = (activation & pianorolls[:, -32:, :]).sum(dim=2)
                union = (activation | pianorolls[:, -32:, :]).sum(dim=2)
                if (union == 0).any():
                    union[union == 0] = 1
                val_accuracy += (intersection / union).sum(dim=1).mean().item() * pianorolls.shape[0]
                val_loss += loss.item() * pianorolls.shape[0]
                val_samples += pianorolls.shape[0]
        val_loss /= val_samples
        train_accuracy /= train_samples
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
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
    # Run the main function with the arguments passed to the script
    main(*sys.argv[1:])