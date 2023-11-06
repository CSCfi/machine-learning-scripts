#!/bin/bash
#SBATCH -p gpusmall --gres=gpu:a100:1 -t 01:00:00 -c 10 
#SBATCH -A project_2001659

module load tensorflow/2.4
module list

set -xv
srun python3 $*
