#!/bin/bash
#SBATCH --job-name=fl_ckks_simulation # [MODIFIED] New job name
#SBATCH --output=/homes/01/sxbhattarai/sec-agg/logs/ckks_simulation_%j.out # [MODIFIED] New output log file
#SBATCH --error=/homes/01/sxbhattarai/sec-agg/logs/ckks_simulation_%j.err   # [MODIFIED] New error log file
#SBATCH --time=23:00:00           # [MODIFIED] SIGNIFICANTLY INCREASED TIME - adjust as needed, might need 24:00:00 or more
#SBATCH --partition=gpu
#SBATCH --nodes=1                 
#SBATCH --cpus-per-task=16        # Keep high for Ray and FHE computations
#SBATCH --gres=gpu:4              # Still useful if clients use GPUs for training

set -e


echo "--- Flower CKKS Simulation Job Started ---" # [MODIFIED]
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
# [MODIFIED] Point to the CKKS main script
/homes/01/sxbhattarai/miniconda3/envs/flwr_env_py310/bin/python "$PROJECT_HOME/main_ckks.py"

echo "--- Flower CKKS Simulation Job Finished ---" # [MODIFIED]


