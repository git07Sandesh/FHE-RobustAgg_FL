from flwr.simulation import start_simulation
from flwr.server import ServerConfig
from sec_agg.client_app import client_fn as raw_client_fn
from sec_agg.server_app import build_strategy
import time
import wandb

# ----------------- Define all experiment configs -----------------
experiment_configs = {
    "krum_plaintext_gauss_f3_n10": {
        "run_name": "krum_plaintext_gauss_f3_n10", "strategy": "Krum", "num_rounds": 20,
        "local_epochs": 1, "alpha": 0.1, "num_malicious": 0, "num_partitions": 10,
        "attack_type": "gaussian_noise", "attack_sigma": 0.1, # A subtle attack
    },
    "krum_plaintext_gauss_f3_n10": {
        "run_name": "krum_plaintext_gauss_f3_n10", "strategy": "Krum", "num_rounds": 20,
        "local_epochs": 1, "alpha": 0.1, "num_malicious": 3, "num_partitions": 10,
        "attack_type": "gaussian_noise", "attack_sigma": 0.1, # A subtle attack
    },
    "krum_plaintext_gauss_f4_n10": {
        "run_name": "krum_plaintext_gauss_f4_n10", "strategy": "Krum", "num_rounds": 20,
        "local_epochs": 1, "alpha": 0.1, "num_malicious": 4, "num_partitions": 10,
        "attack_type": "gaussian_noise", "attack_sigma": 0.1,
    },
}
# To Run individual experiments, uncomment the desired configuration below.
# # ----------------- Select experiment to run -----------------
# FLWR_RUN = "krum_attack_non_iid"  # Change this to run other configs
# run_config = experiment_configs[FLWR_RUN]

# print(f"\n🚀 Running experiment: {FLWR_RUN}\n")

# # ----------------- Client wrapper -----------------
# def client_fn_wrapper(cid: str):
#     partition_id = int(cid)
#     return raw_client_fn(partition_id, run_config["num_partitions"], run_config)

# # ----------------- Build and run strategy -----------------
# strategy = build_strategy(run_config)

# start_time = time.time()
# start_simulation(
#     client_fn=client_fn_wrapper,
#     num_clients=run_config["num_partitions"],
#     config=ServerConfig(num_rounds=run_config["num_rounds"]),
#     strategy=strategy,
# )

# end_time = time.time()
# elapsed_time = end_time - start_time
# wandb.log({"true_runtime_seconds": elapsed_time})

# print(f"⏱️ Total Simulation Time: {elapsed_time:.2f} seconds")


#Use this to run all experiments in the config

# ----------------- Client wrapper -----------------
def client_fn_wrapper_factory(run_config):
    def client_fn_wrapper(cid: str):
        partition_id = int(cid)
        return raw_client_fn(partition_id, run_config["num_partitions"], run_config)
    return client_fn_wrapper

# ----------------- Run all experiments -----------------
for FLWR_RUN, run_config in experiment_configs.items():
    print(f"\n🚀 Starting experiment: {FLWR_RUN}\n")

    # Strategy and client_fn
    strategy = build_strategy(run_config)
    client_fn_wrapper = client_fn_wrapper_factory(run_config)

    # Let the server initialize W&B inside build_strategy/server code

    start_time = time.time()

    start_simulation(
        client_fn=client_fn_wrapper,
        num_clients=run_config["num_partitions"],
        config=ServerConfig(num_rounds=run_config["num_rounds"]),
        strategy=strategy,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Log runtime to existing W&B run (assumes wandb.init() was done in server)
    wandb.log({"true_runtime_seconds": elapsed_time})
    print(f"⏱️ Total Simulation Time for {FLWR_RUN}: {elapsed_time:.2f} seconds")

    # Optional: add delay between runs
    print("🕒 Waiting 30 seconds before next run...\n")
    time.sleep(30)

print("✅ All experiments completed.")