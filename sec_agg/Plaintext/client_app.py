import torch
import numpy as np
import time  # <--- Add this import
from flwr.client import NumPyClient
from .task import SmallNet, load_data, get_weights, set_weights, train

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def client_fn(partition_id: int, num_partitions: int, run_config: dict):
    net = SmallNet().to(DEVICE)
    alpha = run_config.get("alpha", 0.5)
    num_malicious = run_config.get("num_malicious", 0)
    is_byzantine = partition_id < num_malicious
    trainloader, _ = load_data(partition_id, num_partitions, alpha)

    class FlowerClient(NumPyClient):
        def fit(self, parameters, config):
            set_weights(net, parameters)
            
            # --- Start timing the training process ---
            start_time = time.time()
            
            trained_weights = train(net, trainloader, epochs=config["local_epochs"], device=DEVICE)
            
            # --- End timing ---
            end_time = time.time()
            training_time = end_time - start_time
            
            if is_byzantine:
                attack_type = run_config.get("attack_type", "gaussian_noise")
                sigma = run_config.get("attack_sigma", 0.5)
                print(f"[Client {partition_id}] Byzantine client applying '{attack_type}' attack with sigma={sigma}")
                trained_weights = [w + np.random.normal(0, sigma, w.shape).astype(np.float32) for w in trained_weights]
            
            # --- Return training_time in the metrics dictionary ---
            return trained_weights, len(trainloader.dataset), {"training_time": training_time}

    return FlowerClient().to_client()