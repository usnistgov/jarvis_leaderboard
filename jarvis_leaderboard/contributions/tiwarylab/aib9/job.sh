#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=AIB9
#SBATCH -t 48:00:0
#SBATCH -N 1
#SBATCH --gpus=1

module load cuda
module load gcc
module load openmpi
module load gromacs

gmx_mpi grompp -f step5_production.mdp -p topol.top -r step5_1.gro -c step5_1.gro -maxwarn 5 -o pro.tpr
mpirun gmx_mpi mdrun -v -deffnm pro -nb gpu
