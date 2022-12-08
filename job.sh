#!/bin/bash
#SBATCH --partition=preempt --time=1-00:00:00  --output=./out/octobert-med.log 
#SBATCH --mem=175G --gres=gpu:12 -c 20
hostname
date
module load cuda
export CUDA_HOME=/software/cuda/11.3/usr/local/cuda-11.4/bin
source /software/miniconda3/4.12.0/bin/activate CSC298-final
python3 system.py
python3 train_sequence.py --saveTo checkpoints/octobert-med/ --size medium --lakh True --epochs 10