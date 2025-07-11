import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flwr.server import start_server, ServerConfig
from sec_agg.Plaintext.server_app import build_strategy

if __name__ == "__main__":
    run_config = {
        "run_name": "distributed_fedavg_test",
        "strategy": "FedAvg",
        "num_malicious": 0,
        "local_epochs": 3,
        "num_partitions": 10,
        "num_rounds": 5,
        "convergence_threshold": 0.5,
    }

    # Log hostname to confirm placement
    hostname = os.popen("hostname").read().strip()
    print(f"📡 Server running on host: {hostname}")

    # Optional: also write to file for clients to read
    with open("/homes/01/sxbhattarai/sec-agg/server_host.txt", "w") as f:
        f.write(f"{hostname}:8080")

    # This handles wandb.init WITH timeout inside build_strategy
    strategy = build_strategy(run_config)

    # Start Flower server
    start_server(
        server_address="[::]:8080",
        strategy=strategy,
config=ServerConfig(num_rounds=run_config["num_rounds"]),
    )

