#!/bin/bash
#SBATCH --partition=preempt --time=1-00:00:00  --output=./out/infer.log 
#SBATCH --mem=150G --gres=gpu:1
hostname
date
module load cuda
export CUDA_HOME=/software/cuda/11.3/usr/local/cuda-11.4/bin
source /software/miniconda3/4.12.0/bin/activate CSC298-final
python3 system.py
#python3 evaluate.py --name autoencoder-simple-4-13
python3 infer.py