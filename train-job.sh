#!/bin/bash

#SBATCH --mail-user=gmcewan@upei.ca
#SBATCH --mail-type=all

#SBATCH --time=48:00:00
#SBATCH --mem=350G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
python3.6 psoTrain.py -c 32 -m 350 >out.txt 2>err.txt
