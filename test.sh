#!/bin/bash
#SBATCH --partition=standard --time=1-00:00:00  --output=./out/test.log 
#SBATCH --mem=50G -c 20
hostname
date
module load cuda
export CUDA_HOME=/software/cuda/11.3/usr/local/cuda-11.4/bin
source /software/miniconda3/4.12.0/bin/activate CSC298-final
python3 system.py
python3 test.py