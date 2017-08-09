# Machine Learning Applied to Bioinformatics and Speech Technology

Instructions on setting up the second-week project environment for the
[Machine Learning Applied to Bioinformatics and Speech Technology
course](http://www.uef.fi/en/web/summerschool/machine-learning-applied-to-bioinformatics-and-speech-technology)
organized by the Institute of Biomedicine and the School of Computing
of the University of Eastern Finland on August 14-25, 2017.

Every student should have either a CSC account or a temporary training
account.  See
https://research.csc.fi/csc-guide-getting-access-to-csc-services for
instructions on how to register as a new CSC user.

## Logging in and setting up the computing environment

    ssh -l <USERNAME> taito-gpu.csc.fi

    module purge
    module load python-env/3.4.5 cuda/8.0.61 cudnn/6.0
    export THEANO_FLAGS='device=gpu,floatX=float32'
    export OMP_NUM_THREADS=4
    
The `THEANO_FLAGS` environment variable is needed to inform Theano to
use a GPU for computations.  `OMP_NUM_THREADS` limits the number of
CPUs Theano will use.

Run the following commands only once, that is, when setting up the
environment for the first time:

    pip3 install --user /appl/opt/tensorflow/1.2.0/tensorflow-1.2.0-cp34-cp34m-linux_x86_64.whl
    pip3 install --user theano
    pip3 install --user keras
    git clone https://github.com/CSCfi/machine-learning-scripts.git

## Running batch jobs

    cd machine-learning-scripts/slurm
    sbatch run-python-gputest.sh tensorflow-test.py

If you run out of CPU memory, you can increase memory reservation like
this: `--mem=8G`.  Please note that this does not affect GPU memory,
which the K40 and K80 cards have 12 GBs, all of which is automatically
available when the card is reserved.

## Other useful commands

Show all jobs on partition (queue) *gpu*:

    squeue -l -p gpu

Show all own jobs:

    squeue -l -u <USERNAME>

Delete a job:

    scancel <JOBID>

Show an overview of all partitions:

    sinfo
