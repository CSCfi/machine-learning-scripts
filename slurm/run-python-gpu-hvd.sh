#!/bin/bash
#SBATCH -N 1 -n 4 -c 6 -p gpu --gres=gpu:k80:4 -t 1:00:00 --mem=32G

module load python-env/3.6.3-ml
module list

set -xv
mpirun -np 4 -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib -oversubscribe \
    python3.6 $*
