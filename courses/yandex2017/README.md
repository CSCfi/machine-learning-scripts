# Deep neural networks

Instructions on setting up the seminar environment for
the
[Deep neural networks course](https://www.csc.fi/web/training/-/yandex_2017) organized
by [Yandex School of Data Analysis](http://yandexdataschool.com/),
Higher School of Economics ([hse.ru](http://hse.ru)), and CSC on February 1-3, 2017.

Every user should be assigned a unique port number, in this example we
use 8899.  During the course please make sure you use the port number assigned to you by
the course organizers. Otherwise, you are free to select a suitable port from the range 
1024-49151. 

## First terminal window (runs Jupyter notebook server):

    ssh -l USERNAME taito-gpu.csc.fi

    module purge
    module load python-env/3.4.5 cuda/8.0
    export THEANO_FLAGS='device=gpu,floatX=float32'
    export OMP_NUM_THREADS=4
    
The `THEANO_FLAGS` environment variable is needed to inform Theano to use a GPU for computations.  `OMP_NUM_THREADS` limits the number of CPUs Theano will use.

Run the following commands only once, that is, when setting up the environment
for the first time:

    pip3 install --user /appl/opt/tensorflow/0.11.0/tensorflow-0.11.0-py3-none-any.whl
    pip3 install --user --upgrade https://github.com/Theano/Theano/archive/master.zip
    pip3 install --user --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip 
    pip3 install --user notebook


The `srun` command reserves a gpu and opens a shell on one of the compute nodes.  The
option `-t` sets the time limit in the format `HH:MM:SS`, and
`--reservation` gives access to the K80 nodes (4 GPUs per node) reserved for this course
(`XXX` should be either `wed`, `thu` or `fri`; it is also possible to
use the unreserved nodes, by removing the `--reservation` option).

If you run out of CPU memory, you can increase memory reservation like this: 
`--mem-per-cpu=4G`. The reserved nodes have a total of 32 GBs of main memory.
    
    srun -n 1 -p gpu --gres=gpu:1 -t 04:00:00 --reservation=dnn_XXX --mem-per-cpu=4G --pty $SHELL
    hostname  # you need this information later
    .local/bin/jupyter-notebook --no-browser --port=8899

## Second terminal window (for SSH port forwarding):

    ssh -l USERNAME -L 8899:localhost:8899 taito-gpu.csc.fi
    ssh -L 8899:localhost:8899 gXX  # use output of “hostname” command above

## Browser:

Point your browser to the URL given in the first terminal window, e.g.:

http://localhost:8899/?token=c828f3351d0b76ccde12759b942d3ed3c622955e94d6cdc8

# Course repository
https://github.com/yandexdataschool/CSC_deeplearning


