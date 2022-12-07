#!/bin/bash
#SBATCH --partition=preempt --time=1:00:00  --output=./out/aux.log 
#SBATCH -c 20
hostname
date
source /software/miniconda3/4.12.0/bin/activate CSC298-final
cd /home/hliuson/xformers/xformers && pip install -r requirements.txt && pip install -e .