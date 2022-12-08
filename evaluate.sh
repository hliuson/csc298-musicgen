#!/bin/bash
#SBATCH --partition=standard --time=1:00:00  --output=./out/test.log 
#SBATCH --mem=50G -c 20
hostname
date
source /software/miniconda3/4.12.0/bin/activate CSC298-final
python3 system.py
python3 test.py