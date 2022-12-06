#!/bin/bash
#SBATCH --partition=preempt --time=1-00:00:00  --output=./out/lstm-test.log
#SBATCH --mem=170G --gres=gpu:12 -C T4 -c 20
hostname
date
export CUDA_HOME=/software/cuda/11.3/usr/local/cuda-11.4/bin
source /software/miniconda3/4.12.0/bin/activate csc298-musicgen
python3 system.py
#python3 train_autoencoder.py --autoencoder --saveTo checkpoints/autoencoder-simple-16-3/ --epochs 100 --workers 1 --wandbcomment "autoencoder-simple-16-3"
#srun python3 train_autoencoder.py --saveTo checkpoints/convencoder-4pool-4channel-2/ --epochs 100 --workers 1
#python3 train_sequence.py --saveTo checkpoints/sequence-transformer-medium/
srun python3 -u train_sequence.py --workers 4 --epochs 1000 --batch_size 4 --saveTo checkpoints/lstm-test
#srun python3 -u train_sequence.py --workers 4 --epochs 100 --batch_size 4 --saveTo checkpoints/gru-test
#srun python3 -u train_sequence.py --workers 20 --epochs 100 --saveTo checkpoints/tf-test --wandbcomment "tf-test"
