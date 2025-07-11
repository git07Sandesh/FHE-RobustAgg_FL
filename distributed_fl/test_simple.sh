#!/bin/bash
#SBATCH --job-name=simple_test
# [THE MOST IMPORTANT FIX] Use an ABSOLUTE PATH for the log file.
#SBATCH --output=/homes/01/sxbhattarai/sec-agg/logs/simple_%j.out
#SBATCH --error=/homes/01/sxbhattarai/sec-agg/logs/simple_%j.err
#SBATCH --time=00:01:00
#SBATCH --partition=gpu

# We will run the simplest commands possible.
echo "--- Job is running! ---"
echo "Host: $(hostname)"
echo "Working Directory: $(pwd)"
echo "--- Job finished. ---"
