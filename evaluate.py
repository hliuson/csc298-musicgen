from train_autoencoder import *
from autoencoder import *

import torch
import os
import argparse

import sys

def main(*args, **kwargs):
    train, test = download_dataset()
    test_loader = DataLoader(test, batch_size=1, shuffle=True, num_workers=1, collate_fn=MiniCuts())
    #fetch step, epoch from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--name', type=str, default=None)
    
    parse = parser.parse_args(args)
    if parse.step == 0 or parse.epoch == 0:
        print("Error: step and epoch must be specified")
        return

    if parse.name is None:
        print("Error: name must be specified")
        return
    
    #Load model
    #model = ReconstructLossAutoencoder.load_from_checkpoint(f"checkpoints/{parse.name}/epoch={parse.epoch}-step={parse.step}.ckpt")
    #model = SimpleAutoencoder.load_from_checkpoint(f"checkpoints/{parse.name}/epoch={parse.epoch}-step={parse.step}.ckpt")
    model = SimpleAutoencoder.load_from_checkpoint(f"checkpoints/{parse.name}/last.ckpt")
    # Test the model on the test set, and then check iou with different thresholds
    no_examples = 0
    iou = {
        "0.5": 0,
        "0.6": 0,
        "0.7": 0,
        "0.8": 0,
        "0.9": 0,
    }
    
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            reconstruction = model(data)
            for threshold in iou.keys():
                iou[threshold] += iou_score(reconstruction, data, float(threshold))
            no_examples += 1
        
        #print("Test set: Average loss: {:.4f}".format(test_loss))
    for threshold in iou.keys():
        iou[threshold] /= no_examples
        print("Test set: IoU score at threshold {}: {:.4f}".format(threshold, iou[threshold]))

def iou_score(raw, truth, threshold):
    #shape of raw and truth is (batch_size, 32, 128)
    activation = raw > threshold
    truth = truth > 0.5
    intersectionMap = torch.logical_and(activation, truth)
    unionMap = torch.logical_or(activation, truth)
    
    intersections = intersectionMap.sum(dim=(1, 2)) #shape (batch_size)
    unions = unionMap.sum(dim=(1, 2))

    iou = intersections / unions
    #impute 1 for NaNs, because if union is 0 then there are no ground truth positives
    iou = torch.where(torch.isnan(iou), torch.ones_like(iou), iou)
    return iou.mean()
    
if __name__ == '__main__':
    main(*sys.argv[1:])