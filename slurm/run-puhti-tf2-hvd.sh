#!/bin/bash
#SBATCH -N 1 -n 4 -c 10 -p gpu --gres=gpu:v100:4 -t 1:00:00 --mem=64G

module load tensorflow/2.0.0-hvd
module list

set -xv
srun python3.7 $*
