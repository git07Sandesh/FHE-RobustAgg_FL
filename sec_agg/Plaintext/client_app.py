import torch
import numpy as np
import time
from flwr.client import NumPyClient
# [MODIFIED] Import the new helper function
from .task import Net, SmallNet, load_data, get_weights, set_weights, train, get_weights_size_bytes

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def client_fn(partition_id: int, run_config: dict):
    net = SmallNet().to(DEVICE)
    alpha = run_config.get("alpha", 0.5)
    num_malicious = run_config.get("num_malicious", 0)
    is_byzantine = partition_id < num_malicious
    
    # [MODIFIED] Pass partition_id and num_partitions to load_data
    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=run_config["num_partitions"],
        alpha=alpha
    )

    class FlowerClient(NumPyClient):
        def fit(self, parameters, config):
            set_weights(net, parameters)
            
            start_time = time.time()
            trained_weights = train(net, trainloader, epochs=config["local_epochs"], device=DEVICE)
            end_time = time.time()
            training_time = end_time - start_time
            
            if is_byzantine:
                attack_type = run_config.get("attack_type", "gaussian_noise")
                sigma = run_config.get("attack_sigma", 0.5)
                # print(f"[Client {partition_id}] Byzantine client applying '{attack_type}' attack with sigma={sigma}")
                trained_weights = [w + np.random.normal(0, sigma, w.shape).astype(np.float32) for w in trained_weights]
            
            # [NEW] Calculate uplink communication cost
            uplink_bytes = get_weights_size_bytes(trained_weights)
            
            metrics = {
                "training_time": training_time,
                "uplink_bytes": uplink_bytes,
            }
            # For HE schemes, you would add ciphertext size here, e.g.:
            # if "FHE" in run_config["strategy"]:
            #     metrics["ciphertext_size_bytes"] = len(serialized_ciphertext)
            
            return trained_weights, len(trainloader.dataset), metrics

    return FlowerClient().to_client()
