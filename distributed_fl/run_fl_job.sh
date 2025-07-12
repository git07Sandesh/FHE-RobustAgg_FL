#!/bin/bash
#SBATCH --job-name=fl_experiment
#SBATCH --output=/homes/01/sxbhattarai/sec-agg/logs/fl_job_%j.out
#SBATCH --error=/homes/01/sxbhattarai/sec-agg/logs/fl_job_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
# [THE FIX] Request a SINGLE node with all the GPUs on it
#SBATCH --nodes=1
#SBATCH --ntasks=11       # Total number of processes we will launch (1 server + 10 clients)
#SBATCH --gres=gpu:4      # Request all 4 GPUs on a node like gpu002

set -e

echo "--- FL Experiment Job Started on Node: $(hostname) ---"
echo "Job ID: $SLURM_JOB_ID"

# --- Shared Setup ---
export PROJECT_HOME="/homes/01/sxbhattarai/sec-agg"
export PYTHONPATH="$PROJECT_HOME"
export FL_DATA_ROOT="$PROJECT_HOME/data"
export WANDB_MODE=offline
PYTHON_EXE="/homes/01/sxbhattarai/miniconda3/envs/flwr_env_py311/bin/python"
SERVER_HOST_FILE="$PROJECT_HOME/server_host.txt"
rm -f "$SERVER_HOST_FILE"

# --- Start Server in the Background ---
# The server can run on the localhost interface since all clients are on the same node.

SERVER_ADDR="$(hostname):8080"
echo "Starting server, listening on $SERVER_ADDR"
echo "$SERVER_ADDR" > "$SERVER_HOST_FILE"

# Use srun to launch the server as a background task.
# It doesn't need a GPU, so we specify --gpus=0
srun --ntasks=1 --exclusive --gpus=0 \
    "$PYTHON_EXE" "$PROJECT_HOME/distributed_fl/start_server.py" &

# Wait for the server to initialize
echo "Waiting 15 seconds for server to be ready..."
sleep 15

# --- Start Clients in the Background ---
# Use a loop to launch 10 client tasks
for i in {0..9}
do
    echo "Starting client $i..."
    # Use srun to launch each client. Slurm will place them on available resources.
    # We assign GPUs to clients in a round-robin fashion (0, 1, 2, 3, 0, 1, ...)
    # The CUDA_VISIBLE_DEVICES environment variable controls which GPU a process can see.
    GPU_ID=$((i % 4))
    
    # The '&' runs each one in the background.
    srun --ntasks=1 --exclusive --gpus-per-task=1 \
        bash -c "export CUDA_VISIBLE_DEVICES=$GPU_ID; $PYTHON_EXE $PROJECT_HOME/distributed_fl/start_client.py $i" &
done

# --- Wait for all background jobs to finish ---
echo "All processes launched. Waiting for completion..."
wait
echo "--- FL Experiment Job Finished ---"
