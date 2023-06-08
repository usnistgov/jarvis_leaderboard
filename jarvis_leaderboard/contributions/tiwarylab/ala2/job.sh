#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -t 1-0:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="md"

module load gromacs/2020.2-cpu

cpu=$SLURM_NPROCS
sys="AD"
gmx_mpi grompp -f pro.mdp -p ${sys}.top -c NVT_eq.pdb -r NVT_eq.pdb -o pro.tpr -maxwarn 5

mpirun -np $SLURM_NPROCS gmx_mpi mdrun -deffnm md -s pro.tpr -ntomp 8

exit
