#!/bin/bash

#SBATCH --job-name=e269340
#SBATCH --output=../outputs/output_%j.log
#SBATCH --error=../outputs/error_%j.log
#SBATCH --ntasks=10                     # Adjust tasks as needed
#SBATCH --nodes=1                      # Restricted to 1 node
#SBATCH --time=00:15:00                # Partition limit
#SBATCH --partition=577Q               # Explicitly use the default partition

cd ../

#module load gcc/9.3.0
#module load openmpi/4.1.1

mpirun -np $SLURM_NTASKS ./poisson
