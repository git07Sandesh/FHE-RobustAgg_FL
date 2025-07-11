# --- START OF FILE sec_agg/Plaintext/client_app.py ---

import torch
import numpy as np
import time
from flwr.client import NumPyClient
from .task import Net, load_data, get_weights, set_weights, train, get_weights_size_bytes

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# [MODIFIED] The factory now only needs the partition_id
def client_fn(partition_id: int):
    net = Net().to(DEVICE)
    
    # We load data here, but it's static. We'll get dynamic config later.
    # Note: This implies that alpha and num_partitions are part of the dataset setup
    # and don't change per-round, which is a reasonable assumption.
    # We will pass a temporary config just for loading data.
    temp_data_config = {"alpha": 0.5, "num_partitions": 10} # These should match your expected dataset partitions
    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=temp_data_config["num_partitions"],
        alpha=temp_data_config["alpha"]
    )

    class FlowerClient(NumPyClient):
        def fit(self, parameters, config): # <--- `config` is now received from the server
            set_weights(net, parameters)
            
            # [MODIFIED] Use the config received from the server
            is_byzantine = partition_id < config.get("num_malicious", 0)
            
            start_time = time.time()
            # Use local_epochs from the server's config
            trained_weights = train(net, trainloader, epochs=config["local_epochs"], device=DEVICE)
            end_time = time.time()
            training_time = end_time - start_time
            
            if is_byzantine:
                attack_type = config.get("attack_type", "gaussian_noise")
                sigma = config.get("attack_sigma", 0.5)
                trained_weights = [w + np.random.normal(0, sigma, w.shape).astype(np.float32) for w in trained_weights]
            
            uplink_bytes = get_weights_size_bytes(trained_weights)
            
            metrics = {
                "training_time": training_time,
                "uplink_bytes": uplink_bytes,
            }
            
            return trained_weights, len(trainloader.dataset), metrics

    return FlowerClient().to_client()
