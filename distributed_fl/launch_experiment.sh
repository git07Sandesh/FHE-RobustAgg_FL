#!/bin/bash

# Exit immediately if any command fails, preventing silent errors.
set -e

echo "INFO: --- Starting FL Experiment ---"

# This command finds the absolute path to the directory where this script is located.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "INFO: Launching scripts from: $SCRIPT_DIR"

# Define the absolute paths to the job scripts.
SERVER_SCRIPT="$SCRIPT_DIR/run_server.sh"
CLIENT_SCRIPT="$SCRIPT_DIR/run_client.sh"

# --- Submit Server Job ---
echo "INFO: Submitting FL Server job from $SERVER_SCRIPT..."
sbatch "$SERVER_SCRIPT" # We no longer need to capture the Job ID

# [THE FIX] Submit the client job with NO dependency.
# Give the server a few seconds to start up and create the host file.
echo "INFO: Waiting 10 seconds for server to start..."
sleep 10

# --- Submit Client Job Array ---
echo "INFO: Submitting FL Client array job from $CLIENT_SCRIPT..."
sbatch "$CLIENT_SCRIPT"

echo "INFO: --- All jobs submitted. ---"
echo "INFO: Server and clients are now running independently."
echo "INFO: Monitor with 'squeue -u $USER'"
