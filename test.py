from data import *
import torch


def main():
    print("testing")
    train, test = midi_dataset()
    print("train")
    #create a dataloader and sample some data
    dataloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True, num_workers=1, collate_fn=BERTTokenBatcher())
    print("dataloader")
    #sample some data
    batch = next(iter(dataloader))
    midi(batch)
if __name__ == "__main__":
    main()