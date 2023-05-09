. ~/.bashrc
export TMPDIR=/scratch/$SLURM_JOB_ID
cd /wrk/knc6/Software/alignn_calc/jarvis_leaderboard/jarvis_leaderboard/contributions/gpaw_gllbsc
conda activate /wrk/knc6/Software/gpaww
python run.py

