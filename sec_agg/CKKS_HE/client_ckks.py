# sec_agg/client_ckks.py

import torch
import tenseal as ts
import numpy as np
import time
import sys
import pickle

from flwr.client import NumPyClient
from sec_agg.Plaintext.task import Net, SmallNet, get_trainloader, set_weights, train, flatten_weights

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CKKSFlowerClient(NumPyClient):
    def __init__(self, net, trainloader, run_config, context):
        self.net = net
        self.trainloader = trainloader
        self.run_config = run_config
        self.fhe_context = context

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        
        trained_weights = train(self.net, self.trainloader, epochs=config["local_epochs"], device=DEVICE)
        
        partition_id = self.run_config.get("partition_id", -1)
        is_byzantine = partition_id < self.run_config.get("num_malicious", 0)

        if is_byzantine:
            attack_type = self.run_config.get("attack_type", "gaussian_noise")
            sigma = self.run_config.get("attack_sigma", 0.5)
            print(f"[Client {partition_id}] Byzantine client applying '{attack_type}' attack with sigma={sigma}")
            trained_weights = [w + np.random.normal(0, sigma, w.shape).astype(np.float32) for w in trained_weights]

        flat_weights = flatten_weights(trained_weights)
        
        metrics = {}
        strategy_name = self.run_config.get("strategy")
        
        # For TrimmedMean, send the plaintext L2 norm
        if strategy_name == "CKKSTrimmedMean":
            l2_norm = np.linalg.norm(flat_weights)
            metrics["l2_norm"] = float(l2_norm)
            
        # For all schemes, let's log payload sizes for comparison
        plaintext_size = sys.getsizeof(pickle.dumps(trained_weights))
        metrics["plaintext_size_bytes"] = plaintext_size

        # Encrypt and serialize
        encrypted_vector = ts.ckks_vector(self.fhe_context, flat_weights)
        serialized_vector = encrypted_vector.serialize()
        
        encrypted_size = len(serialized_vector)
        metrics["encrypted_size_bytes"] = encrypted_size
        metrics["expansion_factor"] = encrypted_size / plaintext_size

        payload = np.frombuffer(serialized_vector, dtype=np.uint8)
        
        return [payload], len(self.trainloader.dataset), metrics

def client_fn_ckks(partition_id: int, run_config: dict, context_bytes: bytes):
    strategy = run_config.get("strategy")
    net = SmallNet().to(DEVICE)
        
    trainloader = get_trainloader(
        partition_id=partition_id,
        num_partitions=run_config["num_partitions"],
        alpha=run_config["alpha"]
    )
    context = ts.context_from(context_bytes)
    client_run_config = run_config.copy()
    client_run_config["partition_id"] = partition_id
    return CKKSFlowerClient(net, trainloader, client_run_config, context).to_client()