from dataprocess import *
from model import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

### Split the pianorolls into 512-long segments and return a stacked tensor of shape (N, 512, 128),
def get_cuts(pianorolls):
    cuts = []
    for pianoroll in pianorolls:
        for i in range(0, pianoroll.shape[0] - 512, 512):
            cuts.append(torch.Tensor(pianoroll[i : i + 512, :]))
    return torch.stack(cuts)
    

def main():
    dataset = get_dataset()

    ### Train an LSTM model on the MAESTRO dataset to predict the next 32 steps of a pianoroll.

    model = get_baseline_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=get_cuts)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=get_cuts)


    ### Core training loop
    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0
        for pianorolls in train_loader:
            # Move to device
            pianorolls = pianorolls.to(device)
            
            
            outputs= model(pianorolls[:, :-32, :])
            # Get the rest of the 32 steps
            
            
            # Compute loss
            loss = criterion(
                outputs,
                pianorolls[:, 32:, :],
            )
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * pianorolls.shape[0]
        train_loss /= len(train_loader.dataset)
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for pianorolls, _ in val_loader:
                pianorolls = pianorolls.to(device)
                outputs, _ = model(pianorolls[:, :-32, :])
                loss = criterion(
                    outputs.reshape(-1, 128),
                    pianorolls[:, 32:, :].reshape(-1),
                )
                val_loss += loss.item() * pianorolls.shape[0]
        val_loss /= len(val_loader.dataset)
        # Print results
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss = {train_loss:.4f}, "
            f"val_loss = {val_loss:.4f}"
        )

if __name__ == '__main__':
    main()
    