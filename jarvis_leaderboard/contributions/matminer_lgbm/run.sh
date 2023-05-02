#!/bin/bash
#SBATCH --time=59:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=singlegpu
#SBATCH --error=job.err
#SBATCH --output=job.out
. ~/.bashrc
export TMPDIR=/scratch/$SLURM_JOB_ID
conda activate matbench
cd /wrk/knc6/version_tests/tests/ALL_DATASETS/CFID/jarvis_leaderboard/jarvis_leaderboard/benchmarks/matminer_lgbm
python run.py

