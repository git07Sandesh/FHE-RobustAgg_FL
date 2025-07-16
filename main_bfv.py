# main_bfv.py

from flwr.simulation import start_simulation
from flwr.server import ServerConfig
import time
import wandb
import torch  # [NEW] For GPU detection
import ray # [NEW] For explicit Ray shutdown

from sec_agg.BFV_HE.client_bfv import client_fn_bfv
from sec_agg.BFV_HE.server_bfv import build_strategy_bfv

# [NEW] Shutdown any existing Ray instances to prevent resource conflicts
if ray.is_initialized():
    ray.shutdown()
    print("Existing Ray instance shut down before starting simulation.")

# Define a base configuration for reuse
base_config = {
    "num_rounds": 30,  # [MODIFIED] Increased for more meaningful convergence
    "local_epochs": 3,  # [MODIFIED] Increased from 1 to 3
    "alpha": 0.5,
    "num_partitions": 10,
    "attack_type": "gaussian_noise",  # Default attack type
    "attack_sigma": 0.5,  # Default attack intensity
    "num_malicious": 3,  # [NEW] Default number of malicious clients
    "convergence_threshold": 0.50,  # [NEW] Added for convergence tracking
}

# --- Define all BFV FHE experiment configs ---
bfv_experiment_configs = {
    # Benign experiments
    "bfv_fedavg_benign": {
        **base_config, 
        "run_name": "bfv_fedavg_benign", 
        "strategy": "BFVFedAvg", 
        "num_malicious": 0
    },
    "bfv_trimmedmean_benign": {
        **base_config, 
        "run_name": "bfv_trimmedmean_benign", 
        "strategy": "BFVTrimmedMean", 
        "num_malicious": 0
    },

    
    # Attack experiments
   "bfv_fedavg_attack": {
        **base_config, 
        "run_name": "bfv_fedavg_attack", 
        "strategy": "BFVFedAvg"
    },

    "bfv_trimmedmean_attack": {
        **base_config, 
        "run_name": "bfv_trimmedmean_attack", 
        "strategy": "BFVTrimmedMean"
    },

}

# --- Runner script ---
def client_fn_wrapper_factory(run_config, context_bytes):
    def client_fn_wrapper(cid: str):
        return client_fn_bfv(int(cid), run_config, context_bytes)
    return client_fn_wrapper

def main_bfv_runner():  # Wrap in a function
    gpu_available = torch.cuda.is_available()
    # Check if requested GPU resources are within limits (e.g., 10 clients * 0.5 GPU = 5 GPUs needed)
    # Adjust client_resources based on your actual system GPUs and num_partitions
    # For example, if you have 4 GPUs, you might need to set num_gpus to 0.4 or reduce num_partitions.
    num_gpus_per_client = 0.5 # Default. Adjust based on your system and num_partitions
    
    # Example adjustment if 10 clients and 4 GPUs:
    # if gpu_available and base_config["num_partitions"] * num_gpus_per_client > torch.cuda.device_count():
    #     print(f"Warning: Requested GPUs ({base_config['num_partitions']} * {num_gpus_per_client} = {base_config['num_partitions'] * num_gpus_per_client}) exceed available GPUs ({torch.cuda.device_count()}). Adjusting.")
    #     num_gpus_per_client = torch.cuda.device_count() / base_config["num_partitions"]
    #     if num_gpus_per_client < 0.1: # Don't go too low if it's not feasible
    #         print("Calculated num_gpus_per_client is very low, consider reducing num_partitions or not using GPUs.")
    #         num_gpus_per_client = 0.0 # Fallback to CPU if too little GPU per client

    client_resources = {"num_cpus": 2, "num_gpus": num_gpus_per_client if gpu_available else 0.0}
    print(f"🚀 Starting BFV simulations with resources per client: {client_resources}")

    for FLWR_RUN, run_config in bfv_experiment_configs.items():
        print(f"\n🚀 Starting BFV experiment: {FLWR_RUN} using {run_config['strategy']}\n")
        
        # Print attack configuration for clarity
        if run_config["num_malicious"] > 0:
            print(f"   Attack: {run_config['attack_type']} with sigma={run_config['attack_sigma']}")
            print(f"   Malicious clients: {run_config['num_malicious']}/{run_config['num_partitions']}")
        else:
            print("   Benign scenario (no attacks)")
        print()
        
        # The strategy returned is now the MetricsStrategyWrapperBFV
        strategy, context_bytes = build_strategy_bfv(run_config) # Now returns serialized context

        client_fn_wrapper = client_fn_wrapper_factory(run_config, context_bytes)
        
        start_time = time.time()
        start_simulation(
            client_fn=client_fn_wrapper,
            num_clients=run_config["num_partitions"],
            config=ServerConfig(num_rounds=run_config["num_rounds"]),
            strategy=strategy,  # This is the wrapped strategy
            client_resources=client_resources  # Pass client resources
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # The MetricsStrategyWrapperBFV handles wandb.init and logging within its own scope
        # Log total simulation time after the run finishes
        if wandb.run:  # Check if wandb run is active (should be by the wrapper)
            wandb.log({"true_runtime_seconds": elapsed_time})
            strategy.log_final_metrics_and_artifact()  # Call the wrapper's final logging method
            wandb.finish()  # Finish the wandb run
        
        print(f"⏱️ Total BFV Simulation Time for {FLWR_RUN}: {elapsed_time:.2f} seconds")
        print("🕒 Waiting 15 seconds before next run...\n")
        time.sleep(15)

    print("✅ All BFV FHE experiments completed.")
    print("\n💡 Remember to run 'python sec_agg/analysis.py' for plotting both plaintext and BFV results.")

if __name__ == "__main__":
    main_bfv_runner()
