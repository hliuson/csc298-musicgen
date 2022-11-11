#!/bin/bash
#SBATCH --partition=gpu-debug --time=00:30:00  --output=./out/run.log --mem=50G --gres=gpu:2
hostname
date
source /software/miniconda3/4.12.0/bin/activate CSC298-final
python3 train.py --autoencoder --saveTo checkpoints/test-autoencoder.pt --multigpu