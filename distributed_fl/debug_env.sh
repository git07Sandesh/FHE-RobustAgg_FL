#!/bin/bash
#SBATCH --job-name=debug_env
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -e

echo "--- STARTING ENVIRONMENT DEBUG ---"
echo "Running on node: $(hostname)"
echo "User: $(whoami)"

echo "--- CONDA INFO ---"
which conda
conda info --envs

echo "--- INSPECTING flwr_env_py311 ---"
# This command will list all packages and we search for 'flwr'.
# If 'flwr' is not found, grep will have a non-zero exit code and 'set -e' will stop the script.
conda list -n flwr_env_py311 | grep flwr

echo "--- SUCCESS: 'flwr' package was found in the environment. ---"
