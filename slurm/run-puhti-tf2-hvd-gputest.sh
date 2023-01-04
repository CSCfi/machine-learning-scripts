#!/bin/bash
#SBATCH -N 1 -n 4 -c 10 -p gputest --gres=gpu:v100:4 -t 15:00 --mem=64G

module load tensorflow
module list

set -xv
srun python3 $*
