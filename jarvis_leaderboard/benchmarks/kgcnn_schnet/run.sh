#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --partition=singlegpu
#SBATCH --error=job.err
#SBATCH --output=job.out
. ~/.bashrc
export TMPDIR=/scratch/$SLURM_JOB_ID

cd /wrk/knc6/version_tests/tests/jarvis_leaderboard/jarvis_leaderboard/benchmarks/kgcnn_schnet
conda activate /wrk/knc6/Software/kgcnn38
python run.py
