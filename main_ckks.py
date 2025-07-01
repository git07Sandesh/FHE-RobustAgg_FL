# main_ckks.py

from flwr.simulation import start_simulation
from flwr.server import ServerConfig
from sec_agg.CKKS_HE.client_ckks import client_fn_ckks
from sec_agg.CKKS_HE.server_ckks import build_strategy_ckks
import time
import wandb

# Define a base configuration for reuse
base_config = {
    "num_rounds": 2,
    "local_epochs": 1,
    "alpha": 0.5,
    "num_partitions": 10,
    "attack_sigma": 0.2,
}

# --- Define all CKKS FHE experiment configs ---
ckks_experiment_configs = {
    # "ckks_fedavg_benign": {**base_config, "run_name": "ckks_fedavg_benign", "strategy": "CKKSFedAvg", "num_malicious": 0},
    # "ckks_fedavg_attack": {**base_config, "run_name": "ckks_fedavg_attack", "strategy": "CKKSFedAvg", "num_malicious": 3},

    "ckks_krum_benign": {**base_config, "run_name": "ckks_krum_benign", "strategy": "CKKSKrum", "num_malicious": 0},
    "ckks_krum_attack": {**base_config, "run_name": "ckks_krum_attack", "strategy": "CKKSKrum", "num_malicious": 3},

    # "ckks_multikrum_benign": {**base_config, "run_name": "ckks_multikrum_benign", "strategy": "CKKSMultiKrum", "num_malicious": 0},
    # "ckks_multikrum_attack": {**base_config, "run_name": "ckks_multikrum_attack", "strategy": "CKKSMultiKrum", "num_malicious": 3},

    # "ckks_trimmedmean_benign": {**base_config, "run_name": "ckks_trimmedmean_benign", "strategy": "CKKSTrimmedMean", "num_malicious": 0},
    # "ckks_trimmedmean_attack": {**base_config, "run_name": "ckks_trimmedmean_attack", "strategy": "CKKSTrimmedMean", "num_malicious": 3},
}

# --- Runner script ---
def client_fn_wrapper_factory(run_config, context_bytes):
    def client_fn_wrapper(cid: str):
        return client_fn_ckks(int(cid), run_config, context_bytes)
    return client_fn_wrapper

for FLWR_RUN, run_config in ckks_experiment_configs.items():
    print(f"\n🚀 Starting CKKS experiment: {FLWR_RUN} using {run_config['strategy']}\n")
    strategy, context_bytes = build_strategy_ckks(run_config)
    client_fn_wrapper = client_fn_wrapper_factory(run_config, context_bytes)
    
    start_time = time.time()
    start_simulation(
        client_fn=client_fn_wrapper,
        num_clients=run_config["num_partitions"],
        config=ServerConfig(num_rounds=run_config["num_rounds"]),
        strategy=strategy,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    wandb.log({"total_simulation_time": elapsed_time})
    print(f"⏱️ Total CKKS Simulation Time for {FLWR_RUN}: {elapsed_time:.2f} seconds")
    
    wandb.finish()
    print("🕒 Waiting 15 seconds before next run...\n")
    time.sleep(15)

print("✅ All CKKS FHE experiments completed.")