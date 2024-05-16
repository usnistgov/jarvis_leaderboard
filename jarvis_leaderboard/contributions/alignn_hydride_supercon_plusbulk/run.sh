#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=100:00:00
#SBATCH --partition=mml
#SBATCH --error=jobfull.err
#SBATCH --output=jobfull.out


conda activate alignn
train_folder.py --root_dir "data/" --config "config.json" --output_dir=temp-full
