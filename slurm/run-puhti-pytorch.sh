#!/bin/bash
#SBATCH -c 10 -p gpu --gres=gpu:v100:1 -t 1:00:00 --mem=64G

module load pytorch/1.2.0
module list

set -xv
srun python3.7 $*
