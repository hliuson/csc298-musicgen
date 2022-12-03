#!/bin/bash
<<<<<<< HEAD
#SBATCH --partition=preempt --time=1-00:00:00  --output=./out/autoencoder-simple.log 
=======
#SBATCH --partition=preempt --time=1-00:00:00  --output=./out/test.log
>>>>>>> autoencoder
#SBATCH --mem=150G --gres=gpu:12 -C T4 -c 20
hostname
date
source /software/miniconda3/4.12.0/bin/activate CSC298-final
python3 system.py
<<<<<<< HEAD
python3 train.py --autoencoder --saveTo checkpoints/autoencoder-simple-16-3/ --epochs 100 --workers 1 --wandbcomment "autoencoder-simple-16-3"
=======
# python3 train_autoencoder.py --saveTo checkpoints/convencoder-4pool-4channel-2/ --epochs 100 --workers 1
python3 train_sequence.py
>>>>>>> autoencoder
