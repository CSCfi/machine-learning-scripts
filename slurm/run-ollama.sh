#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=15
#SBATCH --gres=gpu:v100:1,nvme:100

# Download ollama models to scratch rather than the home directory
OLLAMA_SCRATCH=/scratch/project_2001659/mvsjober/ollama
export OLLAMA_MODELS=${OLLAMA_SCRATCH}/models

# Add ollama installation dir to PATH
export PATH=/projappl/project_2001659/mvsjober/ollama/bin:$PATH

# Simple way to start ollama. All the server outputs will appear in
# the slurm log mixed with everything else.
#ollama serve &

# If you want to direct ollama server's outputs to a separate log file
# you can start it like this instead
mkdir -p ${OLLAMA_SCRATCH}/logs
ollama serve > ${OLLAMA_SCRATCH}/logs/${SLURM_JOB_ID}.log 2>&1 &

# Capture process id of ollama server
OLLAMA_PID=$!

# Wait to make sure Ollama has started properly
sleep 5

# After this you can use ollama normally in this session

# Example: use ollama commands
ollama pull llama3.1:8b
ollama list
ollama run llama3.1:8b "Why is the sky blue?"

# Example: Try REST API
# curl http://localhost:11434/api/generate -d '{
#   "model": "llama3.1:8b",
#   "prompt":"Why is the sky blue?"
# }'


# At the end of the job, stop the ollama server
kill $OLLAMA_PID
