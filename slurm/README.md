# Example SLURM scripts

Simple SLURM scripts for running Python 3 programs on taito-gpu.csc.fi.  

Usage: `sbatch SHELL-FILE PYTHON-FILE`, e.g. `sbatch run-python-gputest.sh tensorflow-test.py`.

* **run-python-gputest.sh**: SLURM script for running test programs in *gputest* queue
* **run-python-gpu-1h.sh**: SLURM script for running single-gpu programs in *gpu* queue for maximum duration of 1 hour 
* **tensorflow-test.py**: Test program for TensorFlow
* **keras-test.py**: Test program for Keras
* **pytorch-gpu-test.py**: Test program for PyTorch

