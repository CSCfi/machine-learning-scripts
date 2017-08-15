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
    
Note that a specific CSC-compiled version of TensorFlow is required in Taito-GPU. See https://research.csc.fi/-/tensorflow for further instructions.

## Running batch jobs

    cd machine-learning-scripts/slurm
    sbatch run-python-gputest.sh tensorflow-test.py
    sbatch run-python-gpu-1h.sh keras-test.py
    sbatch run-python-gpu-1h.sh keras-mnist_cnn.py
    sbatch run-python-gpu-1h.sh keras-titles-rnn.py

The `gputest` partition is intended for quick testing (as it has a time limit of 15 minutes), so submit all real jobs to `gpu` or `gpulong`.  

The example script `run-python-gpu-1h.sh` has a time limit of 1 hour, which can be changed with the option `-t HOURS:MINUTES:SECONDS` or `-t DAYS-HOURS:MINUTES:SECONDS`.

The example scripts reserve a single GPU with `--gres=gpu:1`.  This can be changed to 2 or 4.  The K40 nodes have 2 GPUs and the K80 nodes have 4 GPUs. 

If you run out of CPU memory, you can increase memory reservation like
this: `--mem=8G`.  Please note that this does not affect GPU memory,
which the K40 and K80 cards have 12 GBs, all of which is automatically
available when the card is reserved.

See `man sbatch` for further information and options.

## Other useful commands

Show all jobs on partition (queue) *gpu*:

    squeue -l -p gpu

Show all own jobs:

    squeue -l -u <USERNAME>

Delete a job:

    scancel <JOBID>

Show an overview of all partitions:

    sinfo

## Data storage

The home directories have a quota of 50 GB and are not intended for research data.  Use `$WRKDIR` for data storage instead. 

See https://research.csc.fi/csc-guide-directories-and-data-storage-at-csc for more information.

## Further information

See the [Taito User Guide](https://research.csc.fi/taito-user-guide), in particular [Section 3: Batch jobs](https://research.csc.fi/taito-batch-jobs) and [Section 6: Using Taito-GPU](https://research.csc.fi/taito-gpu).
