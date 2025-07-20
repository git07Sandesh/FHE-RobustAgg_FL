# main_ckks.py

from flwr.simulation import start_simulation
from flwr.server import ServerConfig
from sec_agg.CKKS_HE.client_ckks import client_fn_ckks
from sec_agg.CKKS_HE.server_ckks import build_strategy_ckks
import time
import wandb
import torch
import itertools # NEW: To chain experiment lists

# Define a base configuration for reuse
# Note: num_malicious and num_partitions here will be overridden by specific experiment configs
base_config = {
    "num_rounds": 150,
    "local_epochs": 3,
    "alpha": 0.5,
    "num_partitions": 10, # Default, will be overridden for scaling
    "attack_sigma": 0.5,
    "num_malicious": 3, # Default, will be overridden for breakpoint and scaling
    "convergence_threshold": 0.50,
    "condition": "attack", # Explicitly set condition
}

# --- 1. Existing CKKS FHE experiment configs (will include the base trimmedmean attack) ---
# We convert this dictionary to a list of its values for easier chaining later
existing_ckks_experiments = [
    {**base_config, "run_name": "ckks_fedavg_benign", "strategy": "CKKSFedAvg", "num_malicious": 0, "condition": "benign"},
    {**base_config, "run_name": "ckks_fedavg_attack", "strategy": "CKKSFedAvg"},

    {**base_config, "run_name": "ckks_krum_benign", "strategy": "CKKSKrum", "num_malicious": 0, "condition": "benign"},
    {**base_config, "run_name": "ckks_krum_attack", "strategy": "CKKSKrum"},

    {**base_config, "run_name": "ckks_multikrum_benign", "strategy": "CKKSMultiKrum", "num_malicious": 0, "condition": "benign"},
    {**base_config, "run_name": "ckks_multikrum_attack", "strategy": "CKKSMultiKrum"},

    {**base_config, "run_name": "ckks_trimmedmean_benign", "strategy": "CKKSTrimmedMean", "num_malicious": 0, "condition": "benign"},
    {**base_config, "run_name": "ckks_trimmedmean_attack", "strategy": "CKKSTrimmedMean"}, # This is the baseline for new experiments
    
    # Bulyan (keeping as per original, though not part of the new focus)
    {**base_config, "run_name": "ckks_bulyan_benign", "strategy": "CKKSBulyan", "num_malicious": 0, "bulyan_selection_size": 8, "trimmed_mean_beta": 1, "condition": "benign"},
    {**base_config, "run_name": "ckks_bulyan_attack", "strategy": "CKKSBulyan", "bulyan_selection_size": 8, "trimmed_mean_beta": 1},
]


# --- 2. Scalability Analysis for CKKSTrimmedMean (under attack) ---
# Goal: Test CKKSTrimmedMean with increasing total clients (N) and proportional malicious clients (f)
# Maintain approx. 30% malicious clients, similar to f=3, N=10 (30%)
#ckks_trimmedmean_attack_scalable_experiments = [
#    # N=10, f=3 (already in existing_ckks_experiments)
#    {**base_config, "run_name": "ckks_trimmedmean_attack_n20_f6", "strategy": "CKKSTrimmedMean", "num_partitions": 20, "num_malicious": 6}, # 30% malicious
#    {**base_config, "run_name": "ckks_trimmedmean_attack_n30_f9", "strategy": "CKKSTrimmedMean", "num_partitions": 30, "num_malicious": 9}, # 30% malicious
#    {**base_config, "run_name": "ckks_trimmedmean_attack_n40_f12", "strategy": "CKKSTrimmedMean", "num_partitions": 40, "num_malicious": 12}, # 30% malicious
#]

# --- 3. Break-Point Analysis for CKKSTrimmedMean (under attack) ---
# Goal: For fixed N=10, increase f until TrimmedMean is expected to fail (N <= 2*f)
# CKKSTrimmedMean works by trimming 'f' clients from each side.
# It requires N > 2*f for proper trimming. If N <= 2*f, it should keep all clients.
#ckks_trimmedmean_attack_breakpoint_experiments = [
#    {**base_config, "run_name": "ckks_trimmedmean_attack_f1_n10", "strategy": "CKKSTrimmedMean", "num_malicious": 1, "num_partitions": 10}, # N=10 > 2*1=2 (OK)
#    {**base_config, "run_name": "ckks_trimmedmean_attack_f2_n10", "strategy": "CKKSTrimmedMean", "num_malicious": 2, "num_partitions": 10}, # N=10 > 2*2=4 (OK)
#    # f=3, N=10 is covered by "ckks_trimmedmean_attack" in existing_ckks_experiments
#    {**base_config, "run_name": "ckks_trimmedmean_attack_f4_n10", "strategy": "CKKSTrimmedMean", "num_malicious": 4, "num_partitions": 10}, # N=10 > 2*4=8 (OK)
#    {**base_config, "run_name": "ckks_trimmedmean_attack_f5_n10", "strategy": "CKKSTrimmedMean", "num_malicious": 5, "num_partitions": 10}, # N=10 <= 2*5=10 (Expected to fallback to keeping all)
#]
#

# --- Combine all experiments for the runner ---
all_ckks_experiments_to_run = list(itertools.chain(
    existing_ckks_experiments,
#    ckks_trimmedmean_attack_scalable_experiments,
#    ckks_trimmedmean_attack_breakpoint_experiments,
))

# --- Runner script ---
def client_fn_wrapper_factory(run_config, context_bytes):
    def client_fn_wrapper(cid: str):
        return client_fn_ckks(int(cid), run_config, context_bytes)
    return client_fn_wrapper

def main_ckks_runner(): # Wrap in a function
    gpu_available = torch.cuda.is_available()
    client_resources = {"num_cpus": 2, "num_gpus": 0.5 if gpu_available else 0.0}
    print(f"🚀 Starting CKKS simulations with resources per client: {client_resources}")

    for run_config in all_ckks_experiments_to_run: # Iterate through the combined list
        current_run_name = run_config["run_name"] # Get the run name from the config dict
        print(f"\n🚀 Starting CKKS experiment: {current_run_name} using {run_config['strategy']}\n")
        
        # The strategy returned is now the MetricsStrategyWrapper
        strategy, context_bytes = build_strategy_ckks(run_config)
        
        client_fn_wrapper = client_fn_wrapper_factory(run_config, context_bytes)
        
        start_time = time.time()
        start_simulation(
            client_fn=client_fn_wrapper,
            num_clients=run_config["num_partitions"],
            config=ServerConfig(num_rounds=run_config["num_rounds"]),
            strategy=strategy, # This is the wrapped strategy
            client_resources=client_resources # Pass client resources
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # The MetricsStrategyWrapperCKKS handles wandb.init and logging within its own scope
        # Log total simulation time after the run finishes
        if wandb.run: # Check if wandb run is active (should be by the wrapper)
            wandb.log({"true_runtime_seconds": elapsed_time})
            strategy.log_final_metrics_and_artifact() # Call the wrapper's final logging method
            wandb.finish() # Finish the wandb run
        
        print(f"⏱️ Total CKKS Simulation Time for {current_run_name}: {elapsed_time:.2f} seconds")
        print("🕒 Waiting 15 seconds before next run...\n")
        time.sleep(15)

    print("✅ All CKKS FHE experiments completed.")
    print("\n💡 Remember to run 'python sec_agg/analysis.py' for plotting both plaintext and CKKS results.")

if __name__ == "__main__":
    main_ckks_runner()
