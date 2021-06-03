# Using Jupyter notebooks on Puhti

This page contains instructions on setting up a Jupyter notebooks server (with optional GPU support) in Puhti. It can be used e.g. to run GPU-enabled TensorFlow, Keras, PyTorch, or Rapids notebooks.

Please note that this is a rather inelegant solution based on using two ssh connections, but can still perhaps be useful in some cases.

In this example, we use 8899 as the port number, but please select a unique port to avoid overlaps.  You are free to select a suitable port from the range 1024-49151.  Overlaps may sometimes still occur, so notice that you might actually get a different port than what you requested.

## First terminal window (runs Jupyter notebook server):

    ssh -l USERNAME puhti.csc.fi

Set up a suitable module environment, for example:

    module purge
    module load tensorflow

More information about data analytics and machine learning environments available on Puhti can be found at https://docs.csc.fi/apps/#data-analytics-and-machine-learning .

The following `srun` command reserves CPUs and GPUs and opens a shell on one of the compute nodes in the `gpu` partition.  The option `-t` sets the time limit in the format `HH:MM:SS`, the option `--mem` sets the memory  reservation, `--gres=gpu:v100:X` reserves `X` GPUs (1<=`X`<=4), and `-c Y` reserves `Y` CPU cores. Replace also `project_xxx` with your compute project.

    srun -A project_xxx -p gpu --gres=gpu:v100:1 -c 10 -t 02:00:00 --mem=64G --pty $SHELL
    hostname  # you need this information later
    jupyter-lab --no-browser --port=8899
    
Note 1: Depending on the Slurm queue, you might have to wait some time before you get the shell access (with the `srun` command). 

Note 2: To run non-GPU notebooks, remove the `--gres` option and change the partition (option `-p`), for example, to `small`.

## Second terminal window (for SSH port forwarding):

In the following commands, replace `8899` with the actual port you got when running the `jupyter-lab` command, and `XX` and `YY` to match the compute node you are using:

    ssh -l USERNAME -L 8899:localhost:8899 puhti.csc.fi
    ssh -L 8899:localhost:8899 rXXgYY  # use output of “hostname” command above

## Browser:

Point your browser to the URL given in the first terminal window, e.g.:

http://localhost:8899/?token=c828f3351d0b76ccde12759b942d3ed3c622955e94d6cdc8
