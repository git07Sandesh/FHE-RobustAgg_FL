# sec_agg/client_paillier.py

import torch
import numpy as np
import time
import pickle
from phe import paillier

from flwr.client import NumPyClient
from sec_agg.Plaintext.task import Net, SmallNet, get_trainloader, set_weights, train, flatten_weights
from .task_paillier import encrypt_weights

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PaillierFlowerClient(NumPyClient):
    def __init__(self, net, trainloader, run_config, public_key):
        self.net = net
        self.trainloader = trainloader
        self.run_config = run_config
        self.public_key = public_key

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        
        trained_weights = train(self.net, self.trainloader, epochs=config["local_epochs"], device=DEVICE)
        
        partition_id = self.run_config.get("partition_id", -1)
        if partition_id < self.run_config.get("num_malicious", 0):
            trained_weights = [-w for w in trained_weights]

        metrics = {}
        strategy_name = self.run_config.get("strategy")
        if strategy_name == "PaillierTrimmedMean":
            # For the leaky protocol, calculate the norm of the *plaintext* weights
            flat_weights = flatten_weights(trained_weights)
            l2_norm = np.linalg.norm(flat_weights)
            metrics["l2_norm"] = float(l2_norm)
            
        # Encrypt the list of weight tensors
        encrypted_update = encrypt_weights(self.public_key, trained_weights)
        
        # Serialize the list of EncryptedNumber arrays using pickle
        payload_bytes = pickle.dumps(encrypted_update)
        payload = np.frombuffer(payload_bytes, dtype=np.uint8)
        
        return [payload], len(self.trainloader.dataset), metrics

def client_fn_paillier(partition_id: int, run_config: dict, paillier_context):
    net = SmallNet().to(DEVICE)
    trainloader = get_trainloader(
        partition_id=partition_id,
        num_partitions=run_config["num_partitions"],
        alpha=run_config["alpha"]
    )
    client_run_config = run_config.copy()
    client_run_config["partition_id"] = partition_id
    
    return PaillierFlowerClient(
        net, trainloader, client_run_config, 
        paillier_context.public_key
    ).to_client()