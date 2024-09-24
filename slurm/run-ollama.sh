#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=15
#SBATCH --gres=gpu:v100:1,nvme:100

# Start ollama server at start of job
export OLLAMA_MODELS=/scratch/project_2001659/mvsjober/ollama-models
mkdir -p $OLLAMA_MODELS

OLLAMA_DIR=/projappl/project_2001659/mvsjober/ollama
export PATH=${OLLAMA_DIR}/bin:$PATH

# Simple way to start ollama. All the server outputs will appear in
# the slurm log mixed with everything else.
#ollama serve &

# If you want to direct ollama server's outputs to a separate log file
# you can start it like this instead:
ollama serve >> ${OLLAMA_DIR}/log 2>&1 &

# After this you can use ollama normally in this session

# Use ollama commands
ollama pull llama3.1:8b
ollama list

# Try REST API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt":"Why is the sky blue?"
}'
