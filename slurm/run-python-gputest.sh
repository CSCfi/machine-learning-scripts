#!/bin/bash
#SBATCH -N 1 -p gputest --gres=gpu:1 -t 15

module list

set -xv

date
hostname
nvidia-smi

python3.4 $*

date
