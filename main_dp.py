# main_dp.py

from flwr.simulation import start_simulation
from flwr.server import ServerConfig
# REUSE the plaintext server logic
from sec_agg.Plaintext.server_app import build_strategy 
# USE the new DP client logic
from sec_agg.DP.client_dp import client_fn_dp 
import time
import wandb

base_config = {
    "num_rounds": 2,
    "local_epochs": 2, # DP often requires more local training
    "alpha": 0.5,
    "num_partitions": 5,
    "num_malicious": 3,
    "dp_max_grad_norm": 1.0,
}

# --- Define all DP experiment configs ---
dp_experiment_configs = {}
# Test different privacy levels (lower epsilon = more privacy = more noise)
for epsilon in [1.0, 2.0, 5.0]:
    for strategy in ["FedAvg", "MultiKrum", "TrimmedMean"]:
        run_name = f"dp_{strategy.lower()}_attack_eps_{epsilon}"
        dp_experiment_configs[run_name] = {
            **base_config, 
            "run_name": run_name, 
            "strategy": strategy,
            "dp_epsilon": epsilon
        }

# --- Runner script ---
def client_fn_wrapper_factory(run_config):
    def client_fn_wrapper(cid: str):
        return client_fn_dp(int(cid), run_config)
    return client_fn_wrapper

for FLWR_RUN, run_config in dp_experiment_configs.items():
    print(f"\n🚀 Starting DP experiment: {FLWR_RUN}\n")
    
    # REUSE the same server_app.py `build_strategy` function
    strategy = build_strategy(run_config)
    client_fn_wrapper = client_fn_wrapper_factory(run_config)
    
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
    print(f"⏱️ Total DP Simulation Time for {FLWR_RUN}: {elapsed_time:.2f} seconds")
    
    wandb.finish()
    time.sleep(15)

print("✅ All DP experiments completed.")