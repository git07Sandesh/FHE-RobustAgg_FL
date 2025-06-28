# sec_agg/client_app.py

import torch
import numpy as np
from flwr.client import NumPyClient
from sec_agg.task import Net, load_data, get_weights, set_weights, train

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def client_fn(partition_id: int, num_partitions: int, run_config: dict):
    net = Net().to(DEVICE)
    alpha = run_config.get("alpha", 0.5)
    num_malicious = run_config.get("num_malicious", 0)
    is_byzantine = partition_id < num_malicious
    trainloader, _ = load_data(partition_id, num_partitions, alpha)

    class FlowerClient(NumPyClient):
        def fit(self, parameters, config):
            set_weights(net, parameters)
            
            # ALL clients train normally first
            trained_weights = train(
                net, trainloader, epochs=config["local_epochs"], device=DEVICE
            )

            # Malicious clients then poison the trained weights
            if is_byzantine:
                attack_type = run_config.get("attack_type", "sign_flip")
                print(f"[Client {partition_id}] Byzantine client - applying '{attack_type}' attack")

                if attack_type == "gaussian_noise":
                    sigma = run_config.get("attack_sigma", 0.1)
                    poisoned_weights = [
                        w + np.random.normal(0, sigma, w.shape).astype(np.float32)
                        for w in trained_weights
                    ]
                    return poisoned_weights, len(trainloader.dataset), {}
                
                # Default to sign-flipping if not specified
                else: # sign_flip
                    return [-w for w in trained_weights], len(trainloader.dataset), {}
            
            # Benign clients return the normal weights
            return trained_weights, len(trainloader.dataset), {}

    return FlowerClient().to_client()