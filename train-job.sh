#!/bin/bash

#SBATCH --mail-user=gmcewan@upei.ca
#SBATCH --mail-type=all

#SBATCH --time=00:20:00
#SBATCH --mem=300G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
python3.6 psoTrain.py -c 32 -m 300 >out.txt 2>err.txt
