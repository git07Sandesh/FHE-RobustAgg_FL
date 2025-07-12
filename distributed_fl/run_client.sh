#!/bin/bash
#SBATCH --job-name=fl_clients
#SBATCH --output=logs/client_%A_%a.out
#SBATCH --error=logs/client_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1  # Request 1 GPU per client task in the array

set -e
# --- Project Setup ---
export PROJECT_HOME="/homes/01/sxbhattarai/sec-agg"
export PYTHONPATH="$PROJECT_HOME"
export FL_DATA_ROOT="$PROJECT_HOME/data"

# Offline settings for HuggingFace
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/homes/01/sxbhattarai/huggingface_cache

# W&B is not needed on the client-side as logging is centralized on the server
# However, it doesn't hurt to leave this here
export WANDB_MODE=offline

# --- Execution ---
echo "INFO: Starting client $SLURM_ARRAY_TASK_ID inside the conda environment..."

# [THE CRITICAL FIX]
conda run -n flwr_env_py311 python "$PROJECT_HOME/distributed_fl/start_client.py" $SLURM_ARRAY_TASK_ID

echo "INFO: Client $SLURM_ARRAY_TASK_ID finished."