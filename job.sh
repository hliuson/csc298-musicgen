#!/bin/bash
#SBATCH --partition=gpu --time=00:30:00  --output=./out/run.log 
#SBATCH --mem=400G --gres=gpu:4 -C A100 -c 50
hostname
date
source /software/miniconda3/4.12.0/bin/activate CSC298-final
python3 train.py --autoencoder --saveTo checkpoints/test-autoencoder.pt