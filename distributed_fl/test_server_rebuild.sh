#!/bin/bash
#SBATCH --job-name=rebuild_test
#SBATCH --output=/homes/01/sxbhattarai/sec-agg/logs/rebuild_%j.out
#SBATCH --error=/homes/01/sxbhattarai/sec-agg/logs/rebuild_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 # Add GPU request back in

set -e

echo "--- Job is running! ---"
echo "Host: $(hostname)"

echo "--- Setting Environment Variables ---"
export PROJECT_HOME="/homes/01/sxbhattarai/sec-agg"
export PYTHONPATH="$PROJECT_HOME"
export FL_DATA_ROOT="$PROJECT_HOME/data"
# ... other env vars ...

echo "--- Attempting Conda Activation ---"
source /homes/01/sxbhattarai/miniconda3/etc/profile.d/conda.sh
conda activate flwr_env_py311
echo "--- Conda activation seems to have worked. ---"

echo "--- Running the Python script ---"
python "$PROJECT_HOME/distributed_fl/start_server.py"

echo "--- Job finished. ---"
