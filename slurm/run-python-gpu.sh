#!/bin/bash
#SBATCH -N 1 -p gpu --gres=gpu:k80:1 -t 1:00:00 --mem=8G

module load python-env/3.6.3-ml
module list

set -xv

date
hostname
nvidia-smi

srun python3.6 $*

date
