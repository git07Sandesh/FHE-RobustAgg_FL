# main_paillier.py

from flwr.simulation import start_simulation
from flwr.server import ServerConfig
import time
import wandb
import torch # For GPU detection
import ray # For explicit Ray shutdown

from sec_agg.Paillier_HE.client_paillier import client_fn_paillier
from sec_agg.Paillier_HE.server_paillier import build_strategy_paillier # This will now return the wrapper

# [NEW] Shutdown any existing Ray instances to prevent resource conflicts
if ray.is_initialized():
    ray.shutdown()
    print("Existing Ray instance shut down before starting simulation.")

base_config = {
    "num_rounds": 30, # [MODIFIED] Increased rounds for better convergence comparison
    "local_epochs": 3, # [MODIFIED] Increased epochs
    "alpha": 0.5,
    "num_partitions": 10,
    "num_malicious": 3,
    "attack_type": "gaussian_noise", # Added for clarity in logs
    "attack_sigma": 0.5, # Added for clarity in logs
    "convergence_threshold": 0.50, # Added for convergence tracking
}

# --- Define Paillier experiment configs ---
paillier_experiment_configs = {
    "paillier_fedavg_benign": {**base_config, "run_name": "paillier_fedavg_benign", "strategy": "PaillierFedAvg", "num_malicious": 0},
    "paillier_fedavg_attack": {**base_config, "run_name": "paillier_fedavg_attack", "strategy": "PaillierFedAvg", "num_malicious": 3},

    "paillier_trimmedmean_benign": {**base_config, "run_name": "paillier_trimmedmean_benign", "strategy": "PaillierTrimmedMean", "num_malicious": 0},
    "paillier_trimmedmean_attack": {**base_config, "run_name": "paillier_trimmedmean_attack", "strategy": "PaillierTrimmedMean", "num_malicious": 3},
}

# --- Runner script ---
def client_fn_wrapper_factory(run_config, paillier_context):
    def client_fn_wrapper(cid: str):
        return client_fn_paillier(int(cid), run_config, paillier_context)
    return client_fn_wrapper

def main_paillier_runner(): # Wrap in a function
    gpu_available = torch.cuda.is_available()
    # Defaulting to 0.5 GPU per client, adjust based on your system and num_partitions
    num_gpus_per_client = 0.5 
    
    # Optional: Dynamic adjustment for GPU allocation if you have fewer total GPUs than needed
    # if gpu_available and base_config["num_partitions"] * num_gpus_per_client > torch.cuda.device_count():
    #     print(f"Warning: Requested GPUs ({base_config['num_partitions']} * {num_gpus_per_client} = {base_config['num_partitions'] * num_gpus_per_client}) exceed available GPUs ({torch.cuda.device_count()}). Adjusting.")
    #     num_gpus_per_client = torch.cuda.device_count() / base_config["num_partitions"]
    #     if num_gpus_per_client < 0.1: 
    #         print("Calculated num_gpus_per_client is very low, considering reducing num_partitions or not using GPUs.")
    #         num_gpus_per_client = 0.0 
    
    client_resources = {"num_cpus": 2, "num_gpus": num_gpus_per_client if gpu_available else 0.0}
    print(f"🚀 Starting Paillier simulations with resources per client: {client_resources}")

    for FLWR_RUN, run_config in paillier_experiment_configs.items():
        print(f"\n🚀 Starting Paillier experiment: {FLWR_RUN} using {run_config['strategy']}\n")
        
        # Print attack configuration for clarity
        if run_config.get("num_malicious", 0) > 0:
            print(f"   Attack: {run_config['attack_type']} with sigma={run_config['attack_sigma']}")
            print(f"   Malicious clients: {run_config['num_malicious']}/{run_config['num_partitions']}")
        else:
            print("   Benign scenario (no attacks)")
        print()
        
        # The build_strategy_paillier function now returns the MetricsStrategyWrapper
        strategy, p_context = build_strategy_paillier(run_config)
        client_fn_wrapper = client_fn_wrapper_factory(run_config, p_context)
        
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
        
        # The MetricsStrategyWrapper handles wandb.init and logging within its own scope
        # Log total simulation time after the run finishes
        if wandb.run: # Check if wandb run is active (should be by the wrapper)
            wandb.log({"true_runtime_seconds": elapsed_time})
            # Call the wrapper's final logging method which also finishes the wandb run
            strategy.log_final_metrics_and_artifact() 
        
        print(f"⏱️ Total Paillier Simulation Time for {FLWR_RUN}: {elapsed_time:.2f} seconds")
        print("🕒 Waiting 15 seconds before next run...\n")
        time.sleep(15)

    print("✅ All Paillier experiments completed.")
    print("\n💡 Remember to run 'python sec_agg/analysis.py' for plotting results.")

if __name__ == "__main__":
    main_paillier_runner() # Call the main function
