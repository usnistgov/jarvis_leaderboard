#!/bin/bash
#SBATCH --time=59:00:00
#SBATCH --mem=190G
#SBATCH --gres=gpu:1
#SBATCH --partition=singlegpu
#SBATCH --error=job10k.err
#SBATCH --output=job10k.out
. ~/.bashrc
export TMPDIR=/scratch/$SLURM_JOB_ID
cd /wrk/knc6/Software/alignn_calc/jarvis_leaderboard/jarvis_leaderboard/contributions/alignn_model/OCP
conda activate /wrk/knc6/Software/alignn_calc
train_folder.py --root_dir "DataDir" --config "tmp_config.json" --output_dir="temp10ka"
