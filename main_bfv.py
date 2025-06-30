# main_bfv.py

from flwr.simulation import start_simulation
from flwr.server import ServerConfig
from sec_agg.BFV_HE.client_bfv import client_fn_bfv
from sec_agg.BFV_HE.server_bfv import build_strategy_bfv
import time
import wandb

base_config = {"num_rounds": 2, "local_epochs": 1, "alpha": 0.5, "num_partitions": 5}

# --- Define all BFV FHE experiment configs ---
bfv_experiment_configs = {
    # "bfv_fedavg_benign": {**base_config, "run_name": "bfv_fedavg_benign", "strategy": "BFVFedAvg", "num_malicious": 0},
    # "bfv_fedavg_attack": {**base_config, "run_name": "bfv_fedavg_attack", "strategy": "BFVFedAvg", "num_malicious": 1},

    # "bfv_multikrum_benign": {**base_config, "run_name": "bfv_multikrum_benign", "strategy": "BFVMultiKrum", "num_malicious": 0},
    # "bfv_multikrum_attack": {**base_config, "run_name": "bfv_multikrum_attack", "strategy": "BFVMultiKrum", "num_malicious": 1},
    
    "bfv_trimmedmean_benign": {**base_config, "run_name": "bfv_trimmedmean_benign", "strategy": "BFVTrimmedMean", "num_malicious": 0},
    "bfv_trimmedmean_attack": {**base_config, "run_name": "bfv_trimmedmean_attack", "strategy": "BFVTrimmedMean", "num_malicious": 1},
}

# --- Runner script ---
def client_fn_wrapper_factory(run_config, context_bytes):
    def client_fn_wrapper(cid: str):
        return client_fn_bfv(int(cid), run_config, context_bytes)
    return client_fn_wrapper

for FLWR_RUN, run_config in bfv_experiment_configs.items():
    print(f"\n🚀 Starting BFV experiment: {FLWR_RUN}")
    strategy, context = build_strategy_bfv(run_config)
    client_fn_wrapper = client_fn_wrapper_factory(run_config, context.serialize(save_secret_key=False))
    
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
    print(f"⏱️ Total BFV Simulation Time for {FLWR_RUN}: {elapsed_time:.2f} seconds")
    
    wandb.finish()
    time.sleep(15)

print("✅ All BFV experiments completed.")