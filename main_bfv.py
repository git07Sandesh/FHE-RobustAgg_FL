# main_bfv.py

from flwr.simulation import start_simulation
from flwr.server import ServerConfig
from sec_agg.BFV_HE.client_bfv import client_fn_bfv
from sec_agg.BFV_HE.server_bfv import build_strategy_bfv
import time
import wandb

base_config = {
    "num_rounds": 10, 
    "local_epochs": 1, 
    "alpha": 0.5, 
    "num_partitions": 10,
    "attack_type": "gaussian_noise",  # Default attack type
    "attack_sigma": 0.5,  # Default attack intensity
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
    # Gaussian noise attack experiments with sigma=0.5
    "bfv_fedavg_gaussian_attack_sigma0.5": {
        **base_config, 
        "run_name": "bfv_fedavg_gaussian_attack_sigma0.5", 
        "strategy": "BFVFedAvg", 
        "num_malicious": 3,
        "attack_type": "gaussian_noise",
        "attack_sigma": 0.5
    },
    "bfv_krum_gaussian_attack_sigma0.5": {
        **base_config, 
        "run_name": "bfv_krum_gaussian_attack_sigma0.5", 
        "strategy": "BFVKrum", 
        "num_malicious": 3,
        "attack_type": "gaussian_noise",
        "attack_sigma": 0.5
    },
    "bfv_multikrum_gaussian_attack_sigma0.5": {
        **base_config, 
        "run_name": "bfv_multikrum_gaussian_attack_sigma0.5", 
        "strategy": "BFVMultiKrum", 
        "num_malicious": 3,
        "attack_type": "gaussian_noise",
        "attack_sigma": 0.5
    },
    "bfv_trimmedmean_gaussian_attack_sigma0.5": {
        **base_config, 
        "run_name": "bfv_trimmedmean_gaussian_attack_sigma0.5", 
        "strategy": "BFVTrimmedMean", 
        "num_malicious": 3,
        "attack_type": "gaussian_noise",
        "attack_sigma": 0.5
    },
    "bfv_bulyan_gaussian_attack_sigma0.5": {
        **base_config, 
        "run_name": "bfv_bulyan_gaussian_attack_sigma0.5", 
        "strategy": "BFVBulyan", 
        "num_malicious": 3,
        "attack_type": "gaussian_noise",
        "attack_sigma": 0.5,
        "bulyan_selection_size": 8,  # Example parameter
        "trimmed_mean_beta": 1,  # Example parameter
    },
}

# --- Runner script ---
def client_fn_wrapper_factory(run_config, context_bytes):
    def client_fn_wrapper(cid: str):
        return client_fn_bfv(int(cid), run_config, context_bytes)
    return client_fn_wrapper

for FLWR_RUN, run_config in bfv_experiment_configs.items():
    print(f"\n🚀 Starting BFV experiment: {FLWR_RUN}")
    
    # Print attack configuration for clarity
    if run_config["num_malicious"] > 0:
        print(f"   Attack: {run_config['attack_type']} with sigma={run_config['attack_sigma']}")
        print(f"   Malicious clients: {run_config['num_malicious']}/{run_config['num_partitions']}")
    else:
        print("   Benign scenario (no attacks)")
    print()
    
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
    
    # Enhanced logging with attack parameters
    wandb.log({
        "total_simulation_time": elapsed_time,
        "attack_type": run_config["attack_type"] if run_config["num_malicious"] > 0 else "none",
        "attack_sigma": run_config["attack_sigma"] if run_config["num_malicious"] > 0 else 0,
        "num_malicious": run_config["num_malicious"],
        "strategy": run_config["strategy"],
    })
    print(f"⏱️ Total BFV Simulation Time for {FLWR_RUN}: {elapsed_time:.2f} seconds")
    
    wandb.finish()
    time.sleep(15)

print("✅ All BFV experiments completed.")