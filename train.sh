#!/bin/sh

#SBATCH -J train.sh
#SBATCH -p titanxp
#SBATCH --gres=gpu:2

srun python3 train.py