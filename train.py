from model import *
from autoencoder import *
import muspy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
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
    for pianoroll in pianorolls:
        for i in range(0, pianoroll.shape[0] - 32, 32):
            cuts.append(torch.Tensor(pianoroll[i : i + 32, :]))
    
    return torch.stack(cuts)

def main(*args, **kwargs):
    #argparse options for new and continue training
    parser = argparse.ArgumentParser()
    parser.add_argument('--continueFrom', type=str, default=None)
    parser.add_argument('--saveTo', type=str, default=None)
    parser.add_argument('--autoencoder', dest='autoencoder', action='store_true')
    parser.add_argument('--LSTM', dest='autoencoder', action='store_false')
    parser.add_argument('--multigpu', dest='multigpu', action='store_true')
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
        wandb.watch(model, log="all", log_freq=10)
        
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
        
    targs = {'epochs': args.epochs, 'model': model, 'optimizer': optimizer, 'saveTo': args.saveTo, 'train': train, 'test': test, 'epoch': epoch, 'batch_size': 4}
    
    if args.multigpu:
        if torch.cuda.device_count() > 1:
            model = nn.DistributedDataParallel(model)
            
            mp.spawn(train_autoencoder, nprocs=torch.cuda.device_count(), args=(targs,))
        else:
            print("WARNING: Multigpu flag set but only one GPU available")
        
    
    
    if args.autoencoder:
        if not args.multigpu:
            train_autoencoder(0, targs)
        else:
            world_size = torch.cuda.device_count()
            mp.spawn(train_autoencoder, nprocs=world_size, args=(world_size, targs,), join=True)
    else:
        train_LSTM(args)

def download_dataset():
    #Load MAESTRO dataset into folder structure
    _ = muspy.datasets.MAESTRODatasetV3(root="data", download_and_extract=True)
    return getdatasets()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def train_autoencoder(rank, world_size, targs):
    model = targs['model']
    optimizer = targs['optimizer']
    saveTo = targs['saveTo']
    train = targs['train']
    test = targs['test']
    epoch = targs['epoch']
    batch_size = targs['batch_size']
    
    if world_size > 1:
        setup(rank, world_size)
        device = torch.device(rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_losses = []
    test_losses = []
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=get_mini_cuts)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=get_mini_cuts)
        
    criterion = nn.MSELoss()
    
    for epoch in range(epoch, epoch + targs['epochs']):
        model.train()
        train_losses = []
        test_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                test_losses.append(loss.item())
        
        print("Epoch: ", epoch, " Train Loss: ", np.mean(train_losses), " Test Loss: ", np.mean(test_losses))
        wandb.log({'train_loss': np.mean(train_losses), 'test_loss': np.mean(test_losses), 'epoch': epoch})
        
        torch.save({
            'model': model,
            'optimizer': optimizer,
            'epoch': epoch,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'run': wandb.run.id
        }, saveTo)
        
    dist.destroy_process_group()
    
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
    
### Empty class which makes passing arguments around easier
class TrainArgs:
    def __init__(self, **entries):
        self.dict = {}
        for key, value in entries.items():
            self.dict[key] = value
    
    def __item__(self, key):
        return self.dict[key]