#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --partition=interactive,singlegpu
#SBATCH --error=job.err
#SBATCH --output=job.out
. ~/.bashrc
export TMPDIR=/scratch/$SLURM_JOB_ID
cd /wrk/knc6/version_tests/tests/ALL_DATASETS/CFID/jarvis_leaderboard/jarvis_leaderboard/benchmarks/resnet_model
conda activate atomvision
date
#pip install atomvision
# https://figshare.com/articles/figure/AtomVision_data/16788268 
train_classifier_cnn.py  --model resnet --train_folder /wrk/knc6/AtomVision/Combined/J2D_C2D_2DMatP/train_folder --test_folder /wrk/knc6/AtomVision/Combined/J2D_C2D_2DMatP/test_folder --epochs 50 --batch_size 16
date
