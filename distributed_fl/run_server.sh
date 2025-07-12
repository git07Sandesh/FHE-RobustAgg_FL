#!/bin/bash
#SBATCH --output=/homes/01/sxbhattarai/sec-agg/logs/server_%j.out
#SBATCH --error=/homes/01/sxbhattarai/sec-agg/logs/server_%j.err
# (other SBATCH directives are correct)
#SBATCH --job-name=fl_server
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

set -e

# --- Project Setup ---
export PROJECT_HOME="/homes/01/sxbhattarai/sec-agg"
export PYTHONPATH="$PROJECT_HOME"
export FL_DATA_ROOT="$PROJECT_HOME/data" # Let's add the data root fix now

# --- Hostname Setup ---
SERVER_HOST_FILE="$PROJECT_HOME/server_host.txt"
rm -f "$SERVER_HOST_FILE"
echo "$(hostname):8080" > "$SERVER_HOST_FILE"

# --- Offline Settings ---
export WANDB_MODE=offline
export WANDB_API_KEY=$(awk '/^machine api.wandb.ai/{getline; print $2}' ~/.netrc)

# --- Execution ---
echo "INFO: Starting the Flower server inside the conda environment..."

# [THE CRITICAL FIX]
# Use 'conda run' to execute the python command directly within the environment.
# This is more reliable for job scripts than 'source' and 'activate'.
conda run -n flwr_env_py311 python "$PROJECT_HOME/distributed_fl/start_server.py"

echo "INFO: Server finished."