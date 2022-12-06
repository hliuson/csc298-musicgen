from data import *
import torch


def main():
    train, test = midi_dataset()
    #create a dataloader and sample some data
    dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    for x in dataloader:
        pitches, durations, positions = x
        print(pitches.shape)
        print(durations.shape)
        print(positions.shape)
        print(pitches)
        print(durations)
        print(positions)
        break

if __name__ == "__main__":
    main()