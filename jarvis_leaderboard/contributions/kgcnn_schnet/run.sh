#! /bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --job-name=schnet
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --output=slurm_%j.output
#SBATCH --partition=gpu_8

ulimit -s unlimited

# Load conda and python
eval "$(conda shell.bash hook)"
echo $CONDA_PREFIX

# Choose environment
conda activate leaderboard
echo $CONDA_PREFIX

# Set path to cuda
# If cudatoolkit in conda path:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# Cuda from modules
module load devel/cuda/11.8
export LD_LIBRARY_PATH=/opt/bwhpc/common/devel/cuda/11.8/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/bwhpc/common/devel/cuda/11.8/
nvidia-smi
echo $LD_LIBRARY_PATH

# Run python script
python3 run.py