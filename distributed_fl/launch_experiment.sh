#!/bin/bash

# Exit immediately if any command fails, preventing silent errors.
set -e

echo "INFO: --- Starting FL Experiment ---"

# This command programmatically finds the absolute path to the directory
# where this script is located. This makes the script portable and robust.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "INFO: Launching scripts from: $SCRIPT_DIR"

# Define the absolute paths to the job scripts, eliminating all ambiguity.
SERVER_SCRIPT="$SCRIPT_DIR/run_server.sh"
CLIENT_SCRIPT="$SCRIPT_DIR/run_client.sh"

# --- Submit Server Job ---
echo "INFO: Submitting FL Server job from $SERVER_SCRIPT..."
SERVER_JOB_ID=$(sbatch --parsable "$SERVER_SCRIPT")
echo "INFO: Server submitted with Job ID: $SERVER_JOB_ID"

# --- Submit Client Job Array ---
echo "INFO: Submitting FL Client array job from $CLIENT_SCRIPT..."
# This job will depend on the successful start of the server job.
sbatch --dependency=afterok:$SERVER_JOB_ID "$CLIENT_SCRIPT"

echo "INFO: --- All jobs submitted. ---"
echo "INFO: Monitor with 'squeue -u $USER'"
echo "INFO: After completion, run 'wandb sync --sync-all' to upload results."