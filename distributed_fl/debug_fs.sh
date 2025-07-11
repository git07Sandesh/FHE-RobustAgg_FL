#!/bin/bash
#SBATCH --job-name=debug_fs
#SBATCH --output=logs/debug_fs_%j.out
#SBATCH --error=logs/debug_fs_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -e

echo "--- STARTING FILESYSTEM DEBUG ---"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"

# Define the path to your environment
CONDA_ENV_PATH="/homes/01/sxbhattarai/miniconda3/envs/flwr_env_py311"
echo "Checking for conda environment at: $CONDA_ENV_PATH"

# Check if the directory exists
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "SUCCESS: Environment directory exists."
else
    echo "FATAL: Environment directory NOT FOUND."
    exit 1
fi

# List some key contents of the site-packages directory
# If this command fails, the directory is not visible or accessible
echo "--- Listing site-packages contents ---"
ls -l "$CONDA_ENV_PATH/lib/python3.11/site-packages/flwr"

echo "--- SUCCESS: 'flwr' package directory found. ---"
