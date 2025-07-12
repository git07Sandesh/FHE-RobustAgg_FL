import time
import wandb
from flwr.simulation import start_simulation
from flwr.server import ServerConfig
import itertools # [NEW] To combine experiment lists

# These imports depend on your project structure.
# Assuming they are correct as in the original code.
from sec_agg.Plaintext.client_app import client_fn as raw_client_fn
from sec_agg.Plaintext.server_app import build_strategy
import torch

# ----------------- Base Configuration -----------------
# Define all common parameters here.
BASE_CONFIG = {
    "num_rounds": 30, # [MODIFIED] Increased for more meaningful convergence
    "local_epochs": 3,
    "alpha": 0.5,
    "num_partitions": 10,
    "num_malicious": 3,
    "attack_type": "gaussian_noise",
    "attack_sigma": 0.5,
    "convergence_threshold": 0.50,
}

# ----------------- Experiment Definitions -----------------
# [MODIFIED] Renamed for clarity
STANDARD_EXPERIMENTS = [
     {"run_name": "fedavg_benign_f0_n10", "strategy": "FedAvg", "num_malicious": 0},
     {"run_name": "fedavg_attack_f3_n10", "strategy": "FedAvg", "num_malicious": 3},
     {"run_name": "krum_benign_f0_n10", "strategy": "Krum", "num_malicious": 0},
     {"run_name": "krum_attack_f3_n10", "strategy": "Krum", "num_malicious": 3},
     {"run_name": "multikrum_benign_f0_n10", "strategy": "MultiKrum", "num_malicious": 0},
     {"run_name": "multikrum_attack_f3_n10", "strategy": "MultiKrum", "num_malicious": 3},
     {"run_name": "trimmedmean_benign_f0_n10", "strategy": "TrimmedMean", "num_malicious": 0},
     {"run_name": "trimmedmean_attack_f3_n10", "strategy": "TrimmedMean", "num_malicious": 3},
     {"run_name": "bulyan_benign_f0_n10","strategy": "Bulyan", "num_malicious": 0,"bulyan_selection_size": 8, "trimmed_mean_beta": 1},
     { "run_name": "bulyan_attack_f3_n10","strategy": "Bulyan", "num_malicious": 3, "bulyan_selection_size": 8, "trimmed_mean_beta": 1},
]

# [NEW] Idea 3: Scalability Analysis
# SCALABILITY_EXPERIMENTS = [
      # Bulyan (n=10, f=3) - Already in STANDARD_EXPERIMENTS
      # FedAvg (n=10, f=3) - Already in STANDARD_EXPERIMENTS
      # n=20, f=6 (~30%)
#      {"run_name": "fedavg_attack_f6_n20", "strategy": "FedAvg", "num_partitions": 20, "num_malicious": 6},
#      {"run_name": "bulyan_attack_f6_n20", "strategy": "Bulyan", "num_partitions": 20, "num_malicious": 6, "bulyan_selection_size": 16, "trimmed_mean_beta": 2},
#      # n=40, f=12 (~30%) - n=50 might be too slow for a quick test
#     # {"run_name": "fedavg_attack_f12_n40", "strategy": "FedAvg", "num_partitions": 40, "num_malicious": 12},
#     # {"run_name": "bulyan_attack_f12_n40", "strategy": "Bulyan", "num_partitions": 40, "num_malicious": 12, "bulyan_selection_size": 28, "trimmed_mean_beta": 4},
# ]

# # [NEW] Idea 4: Break-Point Analysis
# BREAK_POINT_EXPERIMENTS = [
#     # Using n=10, varying f from 1 to 5
#     {"run_name": "fedavg_attack_f1_n10", "strategy": "FedAvg", "num_malicious": 1},
#     {"run_name": "trimmedmean_attack_f1_n10", "strategy": "TrimmedMean", "num_malicious": 1},
#     {"run_name": "bulyan_attack_f1_n10", "strategy": "Bulyan", "num_malicious": 1, "bulyan_selection_size": 9, "trimmed_mean_beta": 1},

#     {"run_name": "fedavg_attack_f2_n10", "strategy": "FedAvg", "num_malicious": 2},
#     {"run_name": "trimmedmean_attack_f2_n10", "strategy": "TrimmedMean", "num_malicious": 2},
#     {"run_name": "bulyan_attack_f2_n10", "strategy": "Bulyan", "num_malicious": 2, "bulyan_selection_size": 8, "trimmed_mean_beta": 1},
    
#     # f=3 is already in STANDARD_EXPERIMENTS
    
#     {"run_name": "fedavg_attack_f4_n10", "strategy": "FedAvg", "num_malicious": 4},
#     {"run_name": "trimmedmean_attack_f4_n10", "strategy": "TrimmedMean", "num_malicious": 4}, # Should be near break-point
#     {"run_name": "bulyan_attack_f4_n10", "strategy": "Bulyan", "num_malicious": 4, "bulyan_selection_size": 6, "trimmed_mean_beta": 1}, # Should also break

#     {"run_name": "fedavg_attack_f5_n10", "strategy": "FedAvg", "num_malicious": 5},
#     {"run_name": "trimmedmean_attack_f5_n10", "strategy": "TrimmedMean", "num_malicious": 5}, # Should break
#     {"run_name": "bulyan_attack_f5_n10", "strategy": "Bulyan", "num_malicious": 5, "bulyan_selection_size": 5, "trimmed_mean_beta": 0}, # Should break
# ]

# ----------------- Client Function Factory -----------------
def get_client_fn(run_config):
    """Factory to create a client_fn that closes over the run_config."""
    def client_fn(cid: str):
        """Flower client function."""
        partition_id = int(cid)
        return raw_client_fn(partition_id, run_config)
    return client_fn

# ----------------- Main Execution Loop -----------------
def main():
    """Run all defined experiments."""
    all_experiments = list(itertools.chain(
        STANDARD_EXPERIMENTS,
        # SCALABILITY_EXPERIMENTS,
        # BREAK_POINT_EXPERIMENTS
    ))

    # [NEW] Determine resources dynamically
    gpu_available = torch.cuda.is_available()
    client_resources = {"num_cpus": 2, "num_gpus": 0.4 if gpu_available else 0.0}
    print(f"🚀 Starting simulations with resources per client: {client_resources}")


    for exp_config in all_experiments:
        # 1. Combine base and experiment-specific configs
        run_config = {**BASE_CONFIG, **exp_config}
        
        run_name = run_config["run_name"]
        print(f"\n🚀 Starting experiment: {run_name}\n")

        # 2. Prepare for the simulation
        # The build_strategy function now returns our MetricsStrategyWrapper
        strategy = build_strategy(run_config)
        client_fn = get_client_fn(run_config)
        
        # 3. Run the simulation and time it
        start_time = time.time()
        start_simulation(
            client_fn=client_fn,
            num_clients=run_config["num_partitions"],
            config=ServerConfig(num_rounds=run_config["num_rounds"]),
            strategy=strategy,
            client_resources=client_resources # Recommended for simulation performance
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 4. [MODIFIED] Log final metrics and clean up
        # The wrapper strategy now handles logging of final summary metrics and artifacts.
        if wandb.run:
            # Log the total wall-clock time for the simulation
            wandb.log({"true_runtime_seconds": elapsed_time})
            
            # Finalize the run by logging summary metrics and artifacts
            strategy.log_final_metrics_and_artifact()
            
            wandb.finish()
            
        print(f"✅ Finished experiment: {run_name}")
        print(f"⏱️ Total Simulation Time: {elapsed_time:.2f} seconds")

        print("🕒 Waiting 5 seconds before next run...\n")
        time.sleep(5)

    print("🎉 All experiments completed.")
    print("\n💡 You can now run 'python sec_agg/analysis.py' to generate comparison plots.")

if __name__ == "__main__":
    main()
