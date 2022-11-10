#!/bin/bash
#SBATCH --partition=gpu-debug --time=00:30:00  --output=./out/run.log --mem=50G
hostname
date
source /software/miniconda3/4.12.0/bin/activate CSC298-final
python3 train.py --autoencoder --saveTo test-autoencoder.pt