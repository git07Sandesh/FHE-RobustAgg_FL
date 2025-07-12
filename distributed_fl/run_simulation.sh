#!/bin/bash
#SBATCH --job-name=fl_simulation
#SBATCH --output=/homes/01/sxbhattarai/sec-agg/logs/simulation_%j.out
#SBATCH --error=/homes/01/sxbhattarai/sec-agg/logs/simulation_%j.err
#SBATCH --time=01:00:00           # Adjust time as needed for your simulation
#SBATCH --partition=gpu
#SBATCH --nodes=1                 # Request a single node
#SBATCH --cpus-per-task=16        # Provide enough CPUs for Ray to manage clients (e.g., 2 CPUs * 10 clients = 20, but 16 is good start)
#SBATCH --gres=gpu:4              # Request all 4 GPUs on a node like gpu002

set -e

echo "--- Flower Simulation Job Started ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"

# --- Set Environment Variables ---
export PROJECT_HOME="/homes/01/sxbhattarai/sec-agg"
export PYTHONPATH="$PROJECT_HOME"
export FL_DATA_ROOT="$PROJECT_HOME/data"
export WANDB_MODE=offline
export WANDB_API_KEY=$(awk '/^machine api.wandb.ai/{getline; print $2}' ~/.netrc)

# --- Execute Main Simulation Script ---
# Use the absolute path to your Python executable
/homes/01/sxbhattarai/miniconda3/envs/flwr_env_py311/bin/python "$PROJECT_HOME/main.py"

echo "--- Flower Simulation Job Finished ---"
