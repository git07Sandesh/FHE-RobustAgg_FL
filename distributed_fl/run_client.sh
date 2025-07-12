#!/bin/bash
#SBATCH --job-name=fl_clients
#SBATCH --output=/homes/01/sxbhattarai/sec-agg/logs/client_%A_%a.out
#SBATCH --error=/homes/01/sxbhattarai/sec-agg/logs/client_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e

echo "--- Client Job is running! ---"
echo "Host: $(hostname)"
echo "Client ID: $SLURM_ARRAY_TASK_ID"

echo "--- Setting Environment Variables ---"
export PROJECT_HOME="/homes/01/sxbhattarai/sec-agg"
export PYTHONPATH="$PROJECT_HOME"
export FL_DATA_ROOT="$PROJECT_HOME/data"
export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
echo "--- Directly executing Python from absolute path ---"
/homes/01/sxbhattarai/miniconda3/envs/flwr_env_py311/bin/python "$PROJECT_HOME/distributed_fl/start_client.py" $SLURM_ARRAY_TASK_ID

echo "--- Client $SLURM_ARRAY_TASK_ID finished. ---"
