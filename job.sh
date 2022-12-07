#!/bin/bash
#SBATCH --partition=preempt --time=1-00:00:00  --output=./out/midibert-1.log
#SBATCH --mem=170G --gres=gpu:4 -c 20
hostname
date
module load cuda
export CUDA_HOME=/software/cuda/11.3/usr/local/cuda-11.4/bin
source /software/miniconda3/4.12.0/bin/activate csc298-musicgen
python3 system.py
srun python3 train_sequence.py --saveTo checkpoints/midibert-1/
