# main_he.py

from flwr.simulation import start_simulation
from flwr.server import ServerConfig
from sec_agg.client_he import client_fn_he
from sec_agg.server_he import build_strategy_he
import time
import wandb

# ----------------- Define all FHE experiment configs -----------------
he_experiment_configs = {

     "he_krum_gauss_f3_n10": {
        "run_name": "he_krum_gauss_f3_n10", "strategy": "HEKrum", "num_rounds": 20,
        "local_epochs": 1, "alpha": 0.1, "num_malicious": 3, "num_partitions": 10,
        "attack_type": "gaussian_noise", "attack_sigma": 0.1,
    },
    "he_krum_gauss_f4_n10": {
        "run_name": "he_krum_gauss_f4_n10", "strategy": "HEKrum", "num_rounds": 20,
        "local_epochs": 1, "alpha": 0.1, "num_malicious": 4, "num_partitions": 10,
        "attack_type": "gaussian_noise", "attack_sigma": 0.1,
    },

}

# ----------------- Client wrapper factory -----------------
def client_fn_wrapper_factory(run_config, context_bytes):
    """Creates a client function that includes the FHE context."""
    def client_fn_wrapper(cid: str):
        partition_id = int(cid)
        return client_fn_he(partition_id, run_config, context_bytes)
    return client_fn_wrapper

# ----------------- Run all experiments -----------------
for FLWR_RUN, run_config in he_experiment_configs.items():
    print(f"\n🚀 Starting FHE experiment: {FLWR_RUN} using {run_config['strategy']}\n")
    # The server builds the strategy and creates the public FHE context
    strategy, context_bytes = build_strategy_he(run_config)
    
    # Create the client factory with the shared context
    client_fn_wrapper = client_fn_wrapper_factory(run_config, context_bytes)
    
    start_time = time.time()

    history = start_simulation(
        client_fn=client_fn_wrapper,
        num_clients=run_config["num_partitions"],
        config=ServerConfig(num_rounds=run_config["num_rounds"]),
        strategy=strategy,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Log total runtime to the same W&B run
    wandb.log({"total_simulation_time": elapsed_time})
    print(f"⏱️ Total FHE Simulation Time for {FLWR_RUN}: {elapsed_time:.2f} seconds")
    
    # Finalize the W&B run
    wandb.finish()

    print("🕒 Waiting 15 seconds before next run...\n")
    time.sleep(15)

print("✅ All FHE experiments completed.")