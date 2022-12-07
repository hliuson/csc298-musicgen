#!/bin/bash
#SBATCH --partition=preempt --time=2-00:00:00  --output=./out/autoencoder_big-300epochs.log 
#SBATCH --mem=150G --gres=gpu:12 -C T4 -c 20
hostname
date
source /software/miniconda3/4.12.0/bin/activate CSC298-final
python3 system.py
python3 train.py --autoencoder --saveTo checkpoints/test-autoencoder-big/ --epochs 300 --workers 1 --loadFrom checkpoints/test-autoencoder-big/last.ckpt