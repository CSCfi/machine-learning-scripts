# Example SLURM scripts

Simple SLURM scripts for running Python 3 programs on taito-gpu.csc.fi.  

## Setting up the environment

Set up the Taito-GPU module environment as:

	module purge
	module load python-env/3.6.3-ml

See [Mlpython](https://research.csc.fi/-/mlpython) for more information on the machine learning modules available for Python in Taito-GPU.

## Running the scripts

Usage: `sbatch SHELL-FILE PYTHON-FILE`, e.g. `sbatch run-python-gputest.sh ../examples/tensorflow-test.py`.

* **run-python-gputest.sh**: SLURM script for running test scripts in *gputest* queue
* **run-python-gpu.sh**: SLURM script for running single-gpu scripts in *gpu* queue
* **run-python-gpu-hvd.sh**: SLURM script for running multi-gpu Horovod scripts in *gpu* queue

The top-level folder `examples` contains several example scripts that can be used, including:

* **tensorflow-test.py**: Test program for TensorFlow
* **keras-test.py**: Test program for Keras
* **keras-mnist_cnn.py**: Classify MNIST digits with a convolutional neural network
* **pytorch-gpu-test.py**: Test program for PyTorch

