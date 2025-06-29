# sec_agg/client_paillier.py

import torch
import numpy as np
import time
import pickle
from phe import paillier

from flwr.client import NumPyClient
from sec_agg.Plaintext.task import Net, get_trainloader, set_weights, train, flatten_weights
from .task_paillier import encode_weights, encrypt_vector

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PaillierFlowerClient(NumPyClient):
    def __init__(self, net, trainloader, run_config, public_key, precision):
        self.net = net
        self.trainloader = trainloader
        self.run_config = run_config
        self.public_key = public_key
        self.precision = precision

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        
        trained_weights = train(self.net, self.trainloader, epochs=config["local_epochs"], device=DEVICE)
        
        partition_id = self.run_config.get("partition_id", -1)
        if partition_id < self.run_config.get("num_malicious", 0):
             # Apply attack
            trained_weights = [-w for w in trained_weights]

        flat_weights = flatten_weights(trained_weights)
        
        # 1. Encode floats to integers
        encoded_weights = encode_weights(flat_weights, self.precision)
        
        # 2. Encrypt the integer vector
        encrypted_weights = encrypt_vector(self.public_key, encoded_weights)
        
        metrics = {}
        strategy_name = self.run_config.get("strategy")
        if strategy_name == "PaillierTrimmedMean":
            l2_norm = np.linalg.norm(flat_weights) # Use norm of original float weights
            metrics["l2_norm"] = float(l2_norm)
            
        # 3. Serialize the list of EncryptedNumber objects using pickle
        payload_bytes = pickle.dumps(encrypted_weights)
        payload = np.frombuffer(payload_bytes, dtype=np.uint8)
        
        return [payload], len(self.trainloader.dataset), metrics

def client_fn_paillier(partition_id: int, run_config: dict, paillier_context):
    net = Net().to(DEVICE)
    trainloader = get_trainloader(
        partition_id=partition_id,
        num_partitions=run_config["num_partitions"],
        alpha=run_config["alpha"]
    )
    client_run_config = run_config.copy()
    client_run_config["partition_id"] = partition_id
    
    return PaillierFlowerClient(
        net, trainloader, client_run_config, 
        paillier_context.public_key, paillier_context.precision
    ).to_client()