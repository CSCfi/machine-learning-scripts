# Using Jupyter notebooks on Taito-GPU

Instructions on setting up a Jupyter notebooks server with TensorFlow, Keras, and GPU support in Taito-GPU.
This is a rather inelegant solution based on using two ssh connections, but can still perhaps be useful in some cases.

In this example, we use 8899 as the port number, but please select a unique port to avoid 
overlaps.  You are free to select a suitable port from the range 1024-49151. 

## First terminal window (runs Jupyter notebook server):

    ssh -l USERNAME taito-gpu.csc.fi

    module purge
    module load python-env/3.4.5 cuda/8.0.61 cudnn/6.0

Run the following commands only once, that is, when setting up the environment
for the first time:

    pip3 install --user /appl/opt/tensorflow/1.3.0/tensorflow-1.3.0-cp34-cp34m-linux_x86_64.whl
    pip3 install --user keras notebook

Further instructions on setting up TensorFlow in Taito-GPU can be found at https://research.csc.fi/-/tensorflow .

The `srun` command reserves a gpu and opens a shell on one of the compute nodes.  The
option `-t` sets the time limit in the format `HH:MM:SS`, the option `--mem` sets the memory 
reservation, and `--gres=gpu:X` reserves `X` GPUs (1<=`X`<=4).

    srun -p gpu --gres=gpu:1 -t 04:00:00 --mem=8G --pty $SHELL
    hostname  # you need this information later
    .local/bin/jupyter-notebook --no-browser --port=8899

## Second terminal window (for SSH port forwarding):

    ssh -l USERNAME -L 8899:localhost:8899 taito-gpu.csc.fi
    ssh -L 8899:localhost:8899 gXX  # use output of “hostname” command above

## Browser:

Point your browser to the URL given in the first terminal window, e.g.:

http://localhost:8899/?token=c828f3351d0b76ccde12759b942d3ed3c622955e94d6cdc8
