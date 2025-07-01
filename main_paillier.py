# main_paillier.py

from flwr.simulation import start_simulation
from flwr.server import ServerConfig
from sec_agg.Paillier_HE.client_paillier import client_fn_paillier
from sec_agg.Paillier_HE.server_paillier import build_strategy_paillier
import time
import wandb

base_config = {
    "num_rounds": 2,
    "local_epochs": 1,
    "alpha": 0.5,
    "num_partitions": 5, # Starting with 5 to keep it fast
}

# --- Define Paillier experiment configs ---
paillier_experiment_configs = {
    # "paillier_fedavg_benign": {**base_config, "run_name": "paillier_fedavg_benign", "strategy": "PaillierFedAvg", "num_malicious": 0},
    # "paillier_fedavg_attack": {**base_config, "run_name": "paillier_fedavg_attack", "strategy": "PaillierFedAvg", "num_malicious": 1},

    # "paillier_trimmedmean_benign": {**base_config, "run_name": "paillier_trimmedmean_benign", "strategy": "PaillierTrimmedMean", "num_malicious": 0},
    # "paillier_trimmedmean_attack": {**base_config, "run_name": "paillier_trimmedmean_attack", "strategy": "PaillierTrimmedMean", "num_malicious": 1},
    "paillier_krum_benign": {**base_config, "run_name": "paillier_krum_benign", "strategy": "PaillierKrum", "num_malicious": 0},
    "paillier_krum_attack": {**base_config, "run_name": "paillier_krum_attack", "strategy": "PaillierKrum", "num_malicious": 1},
}

# --- Runner script ---
def client_fn_wrapper_factory(run_config, paillier_context):
    def client_fn_wrapper(cid: str):
        return client_fn_paillier(int(cid), run_config, paillier_context)
    return client_fn_wrapper

for FLWR_RUN, run_config in paillier_experiment_configs.items():
    print(f"\n🚀 Starting Paillier experiment: {FLWR_RUN}")
    strategy, p_context = build_strategy_paillier(run_config)
    client_fn_wrapper = client_fn_wrapper_factory(run_config, p_context)
    
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
    print(f"⏱️ Total Paillier Simulation Time for {FLWR_RUN}: {elapsed_time:.2f} seconds")
    
    wandb.finish()
    time.sleep(15)

print("✅ All Paillier experiments completed.")