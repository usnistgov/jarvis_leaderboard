#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=100:00:00
#SBATCH --partition=mml
#SBATCH --error=job3.err
#SBATCH --output=job3.out


conda activate alignn
train_folder.py --root_dir "data/" --config "config.json" --output_dir=temp-3
