#!/bin/bash

#SBATCH --mail-user=gmcewan@upei.ca
#SBATCH --mail-type=all

#SBATCH --time=00:20:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
python3.6 psoTrain.py -c 5 -m 10 >out.txt 2>err.txt
