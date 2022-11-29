#!/bin/bash
#SBATCH --partition=preempt --time=1-00:00:00  --output=./out/autoencoder-simple-eval.log 
#SBATCH --mem=150G --gres=gpu:12 -C T4 -c 20
hostname
date
source /software/miniconda3/4.12.0/bin/activate CSC298-final
python3 system.py
python3 evaluate.py --name autoencoder-simple-16-3 --epoch 2 --step 288