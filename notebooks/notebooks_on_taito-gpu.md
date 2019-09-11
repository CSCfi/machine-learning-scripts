# Using Jupyter notebooks on Taito-GPU

This page contains instructions on setting up a Jupyter notebooks server with GPU support in Taito-GPU. It can be used e.g. to run GPU-enabled TensorFlow, Keras, or PyTorch notebooks.

Please note that this is a rather inelegant solution based on using two ssh connections, but can still perhaps be useful in some cases.

In this example, we use 8899 as the port number, but please select a unique port to avoid 
overlaps.  You are free to select a suitable port from the range 1024-49151. 

## First terminal window (runs Jupyter notebook server):

    ssh -l USERNAME taito-gpu.csc.fi

Set up a suitable module environment, for example:

    module purge
    module load python-env/3.6.3-ml

Further instructions on setting up TensorFlow in Taito-GPU can be found at https://research.csc.fi/-/tensorflow . More information about Mlpython (machine learning for Python) environments can be found at https://research.csc.fi/-/mlpython .

The following `srun` command reserves CPUs and GPUs and opens a shell on one of the compute nodes.  The
option `-t` sets the time limit in the format `HH:MM:SS`, the option `--mem` sets the memory 
reservation, `--gres=gpu:p100:X` reserves `X` (Pascal P100) GPUs (1<=`X`<=4), and `-c Y` reserves `Y` CPU cores.

    srun -p gpu --gres=gpu:p100:1 -c 4 -t 04:00:00 --mem=8G --pty $SHELL
    hostname  # you need this information later
    jupyter-notebook --no-browser --port=8899

## Second terminal window (for SSH port forwarding):

    ssh -l USERNAME -L 8899:localhost:8899 taito-gpu.csc.fi
    ssh -L 8899:localhost:8899 gXX  # use output of “hostname” command above

## Browser:

Point your browser to the URL given in the first terminal window, e.g.:

http://localhost:8899/?token=c828f3351d0b76ccde12759b942d3ed3c622955e94d6cdc8
