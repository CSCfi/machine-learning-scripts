#!/bin/bash
#SBATCH -N 1 -p gpu --gres=gpu:1 -t 1:00:00

module list

set -xv

date
hostname
nvidia-smi

python3.4 $*

date
