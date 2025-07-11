#!/bin/bash
#SBATCH --job-name=fl_server
#SBATCH --output=/homes/01/sxbhattarai/sec-agg/logs/server_%j.out
#SBATCH --error=/homes/01/sxbhattarai/sec-agg/logs/server_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e

echo "--- Job is running! ---"
echo "Host: $(hostname)"

echo "--- Setting Environment Variables ---"
export PROJECT_HOME="/homes/01/sxbhattarai/sec-agg"
export PYTHONPATH="$PROJECT_HOME"
export FL_DATA_ROOT="$PROJECT_HOME/data"
export WANDB_MODE=offline

# --- Hostname Setup ---
SERVER_HOST_FILE="$PROJECT_HOME/server_host.txt"
rm -f "$SERVER_HOST_FILE"
echo "$(hostname):8080" > "$SERVER_HOST_FILE"

echo "--- Directly executing Python from absolute path ---"
# This bypasses any and all conda activation issues.
/homes/01/sxbhattarai/miniconda3/envs/flwr_env_py311/bin/python "$PROJECT_HOME/distributed_fl/start_server.py"

echo "--- Job finished. ---"
