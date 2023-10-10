#!/bin/bash
#SBATCH --account=<project>
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=1:00:00

module list
set -xv

srun python3 $*
