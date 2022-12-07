#!/bin/bash
#SBATCH --partition=standard --time=1-00:00:00  --output=./out/download.log 
#SBATCH --mem=150G -n 20 -c 20
hostname
date
source /software/miniconda3/4.12.0/bin/activate csc298-musicgen
python3 system.py
python3 clean.py
