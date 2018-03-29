# Example SLURM scripts

Simple SLURM scripts for running Python 3 programs on taito-gpu.csc.fi.  

## Setting up the environment

Set up the Taito-GPU module environment as:

	module purge
	module load python-env/3.5.3-ml

See [Mlpython](https://research.csc.fi/-/mlpython) for more information on the machine learning modules available for Python in Taito-GPU.

## Running the scripts

Usage: `sbatch SHELL-FILE PYTHON-FILE`, e.g. `sbatch run-python-gputest.sh tensorflow-test.py`.

* **run-python-gputest.sh**: SLURM script for running test programs in *gputest* queue
* **run-python-gpu-1h.sh**: SLURM script for running single-gpu programs in *gpu* queue for maximum duration of 1 hour 
* **tensorflow-test.py**: Test program for TensorFlow
* **keras-test.py**: Test program for Keras
* **keras-mnist_cnn.py**: Classify MNIST digits with a convolutional neural network
* **keras-titles-rnn.py**: Generate book and movie titles with a recurrent neural network
* **pytorch-gpu-test.py**: Test program for PyTorch

