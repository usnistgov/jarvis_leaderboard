#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=interactive,general,batch,singlegpu
#SBATCH --error=job.err
#SBATCH --output=job.out
. ~/.bashrc
export TMPDIR=/scratch/$SLURM_JOB_ID
conda activate /wrk/knc6/matformer
cd /wrk/knc6/version_tests/tests/ALL_DATASETS/CFID/jarvis_leaderboard/jarvis_leaderboard/benchmarks/matformer
python run.py

