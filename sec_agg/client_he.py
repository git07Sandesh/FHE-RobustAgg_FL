
import torch
import tenseal as ts
import numpy as np
import time
from flwr.client import NumPyClient
from .task import Net, SmallNet, get_trainloader, set_weights, train, flatten_weights

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HEFlowerClient(NumPyClient):
    def __init__(self, net, trainloader, run_config, context):
        self.net = net
        self.trainloader = trainloader
        self.run_config = run_config
        self.fhe_context = context

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        
        partition_id = self.run_config.get("partition_id", -1)
        num_malicious = self.run_config.get("num_malicious", 0)
        is_byzantine = partition_id < num_malicious

        trained_weights = train(
            self.net, self.trainloader, epochs=config["local_epochs"], device=DEVICE
        )
        
        if is_byzantine:
            print(f"[Client {partition_id}] Byzantine client - flipping weights BEFORE encryption")
            trained_weights = [-w for w in trained_weights]
        
        flat_weights = flatten_weights(trained_weights)
        
        print(f"[Client {partition_id}] Encrypting weights...")
        encryption_start_time = time.time()
        encrypted_vector = ts.ckks_vector(self.fhe_context, flat_weights)
        
        serialized_vector = encrypted_vector.serialize()
        encryption_duration = time.time() - encryption_start_time
        print(f"[Client {partition_id}] Encryption took {encryption_duration:.4f}s")
        
        # FIXED: Convert bytes to a NumPy array of uint8 for safe transport
        payload = np.frombuffer(serialized_vector, dtype=np.uint8)
        
        return [payload], len(self.trainloader.dataset), {"encryption_time": encryption_duration}

# (The rest of client_he.py remains the same)
def client_fn_he(partition_id: int, run_config: dict, context_bytes: bytes):
    strategy = run_config.get("strategy", "HEFedAvg")
    net = (SmallNet() if strategy == "HEKrum" else Net()).to(DEVICE)
    trainloader = get_trainloader(
        partition_id=partition_id,
        num_partitions=run_config["num_partitions"],
        alpha=run_config["alpha"]
    )
    context = ts.context_from(context_bytes)
    client_run_config = run_config.copy()
    client_run_config["partition_id"] = partition_id
    return HEFlowerClient(net, trainloader, client_run_config, context).to_client()