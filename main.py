import time
import wandb
from flwr.simulation import start_simulation
from flwr.server import ServerConfig

# These imports depend on your project structure.
# Assuming they are correct as in the original code.
from sec_agg.Plaintext.client_app import client_fn as raw_client_fn
from sec_agg.Plaintext.server_app import build_strategy

# ----------------- Base Configuration -----------------
# Define all common parameters here.
BASE_CONFIG = {
    "num_rounds": 10,
    "local_epochs": 1,
    "alpha": 0.5,
    "num_partitions": 10,
    "num_malicious": 3,  # Default number of malicious clients for attack scenarios
    "attack_type": "gaussian_noise",  # Default attack type
    "attack_sigma": 0.5
}

# ----------------- Experiment Definitions -----------------
# Each dictionary only needs to specify what's different from the BASE_CONFIG.
EXPERIMENTS = [
    {
        "run_name": "fedavg_benign_f3_n10",
        "strategy": "FedAvg",
        "num_malicious": 0,  # Override the base value for the benign case
    },
    {
        "run_name": "fedavg_attack_f3_n10",
        "strategy": "FedAvg",
        # Uses num_malicious=3 from BASE_CONFIG
    },
    {
        "run_name": "krum_attack_f3_n10",
        "strategy": "Krum",
    },
    {
        "run_name": "multikrum_attack_f3_n10",
        "strategy": "MultiKrum",
    },
    {
        "run_name": "trimmedmean_attack_f3_n10",
        "strategy": "TrimmedMean",
    },
    {
        "run_name": "bulyan_attack_f3_n10",
        "strategy": "Bulyan",
        # Bulyan-specific parameters are added here
        "bulyan_selection_size": 7,
        "trimmed_mean_beta": 1,
    },
]

# ----------------- Client Function Factory -----------------
def get_client_fn(run_config):
    """Factory to create a client_fn that closes over the run_config."""
    def client_fn(cid: str):
        """Flower client function."""
        partition_id = int(cid)
        return raw_client_fn(partition_id, run_config["num_partitions"], run_config)
    return client_fn

# ----------------- Main Execution Loop -----------------
def main():
    """Run all defined experiments."""
    for exp_config in EXPERIMENTS:
        # 1. Combine base and experiment-specific configs
        # The `**exp_config` will override any keys from `**BASE_CONFIG`
        run_config = {**BASE_CONFIG, **exp_config}
        
        run_name = run_config["run_name"]
        print(f"\n🚀 Starting experiment: {run_name}\n")

        # 2. Prepare for the simulation
        # It's assumed that `build_strategy` will initialize wandb.
        strategy = build_strategy(run_config)
        client_fn = get_client_fn(run_config)
        
        # 3. Run the simulation and time it
        start_time = time.time()
        start_simulation(
            client_fn=client_fn,
            num_clients=run_config["num_partitions"],
            config=ServerConfig(num_rounds=run_config["num_rounds"]),
            strategy=strategy,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 4. Log final metrics and clean up
        # This assumes the wandb run is still active from the strategy initialization
        if wandb.run:
            wandb.log({"true_runtime_seconds": elapsed_time})
            wandb.finish() # End the current W&B run before starting the next
            
        print(f"✅ Finished experiment: {run_name}")
        print(f"⏱️ Total Simulation Time: {elapsed_time:.2f} seconds")

        # Optional: add a delay between runs to prevent rate-limiting or other issues
        print("🕒 Waiting 10 seconds before next run...\n")
        time.sleep(10)

    print("🎉 All experiments completed.")

if __name__ == "__main__":
    main()