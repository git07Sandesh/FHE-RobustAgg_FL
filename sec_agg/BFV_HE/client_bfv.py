# sec_agg/client_bfv.py

import torch
import tenseal as ts
import numpy as np
from flwr.client import NumPyClient
from sec_agg.Plaintext.task import Net, SmallNet, get_trainloader, set_weights, train, flatten_weights
from .task_bfv import encode

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BFVFlowerClient(NumPyClient):
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
        
        # 1. Encode floats to integers using fixed-point representation
        encoded_vector = encode(flat_weights)
        
        # 2. Encrypt the integer vector with BFV
        encrypted_vector = ts.bfv_vector(self.fhe_context, encoded_vector)
        
        # 3. Serialize and send
        serialized_vector = encrypted_vector.serialize()
        payload = np.frombuffer(serialized_vector, dtype=np.uint8)
        
        return [payload], len(self.trainloader.dataset), {}

def client_fn_bfv(partition_id: int, run_config: dict, context_bytes: bytes):
    strategy = run_config.get("strategy")
    net = SmallNet().to(DEVICE)
        
    trainloader = get_trainloader(partition_id=partition_id, num_partitions=run_config["num_partitions"], alpha=run_config["alpha"])
    context = ts.context_from(context_bytes)
    client_run_config = run_config.copy()
    client_run_config["partition_id"] = partition_id
    
    return BFVFlowerClient(net, trainloader, client_run_config, context).to_client()