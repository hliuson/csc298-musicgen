#!/bin/bash
#SBATCH --partition=preempt --time=1-00:00:00  --output=./out/sequence-transformer-medium.log 
#SBATCH --mem=150G --gres=gpu:12 -C T4 -c 20
hostname
date
export CUDA_HOME=/software/cuda/11.3/usr/local/cuda-11.4/bin
source /software/miniconda3/4.12.0/bin/activate CSC298-final
python3 system.py
#python3 train_autoencoder.py --saveTo checkpoints/autoencoder-simple-mlp-dropout-8-13-128/ --epochs 100 --workers 1
python3 train_sequence.py --saveTo checkpoints/sequence-transformer-medium/