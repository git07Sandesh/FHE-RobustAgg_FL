# sec_agg/client_bfv.py

import torch
import tenseal as ts
import numpy as np
import time
import sys
import pickle

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
        
        train_start_time = time.perf_counter() # Use perf_counter
        trained_weights = train(self.net, self.trainloader, epochs=config["local_epochs"], device=DEVICE)
        train_end_time = time.perf_counter()
        training_time = train_end_time - train_start_time
        
        partition_id = self.run_config.get("partition_id", -1)
        is_byzantine = partition_id < self.run_config.get("num_malicious", 0)

        if is_byzantine:
            attack_type = self.run_config.get("attack_type", "gaussian_noise")
            sigma = self.run_config.get("attack_sigma", 0.5)
            print(f"[Client {partition_id}] Byzantine client applying '{attack_type}' attack with sigma={sigma}")
            trained_weights = [w + np.random.normal(0, sigma, w.shape).astype(np.float32) for w in trained_weights]

        flat_weights = flatten_weights(trained_weights)

        metrics = {
            "training_time": training_time,
        }
        
        strategy_name = self.run_config.get("strategy")
        
        # For TrimmedMean/Bulyan, send the plaintext L2 norm
        if strategy_name in ["BFVTrimmedMean", "BFVBulyan"]: # Added Bulyan
            l2_norm = np.linalg.norm(flat_weights)
            metrics["l2_norm"] = float(l2_norm)
            
        # For all schemes, let's log payload sizes for comparison
        # Important: Calculate plaintext_size *before* encryption/serialization for accurate expansion factor
        plaintext_payload = pickle.dumps(trained_weights)
        plaintext_size = sys.getsizeof(plaintext_payload)
        metrics["plaintext_size_bytes"] = plaintext_size
        
        # 1. Encode floats to integers using fixed-point representation
        encoded_vector = encode(flat_weights)
        
        # 2. Encrypt the integer vector with BFV
        enc_start = time.perf_counter() # Measure encryption time
        encrypted_vector = ts.bfv_vector(self.fhe_context, encoded_vector)
        
        # 3. Serialize and send
        serialized_vector = encrypted_vector.serialize()
        enc_end = time.perf_counter()
        encryption_time = enc_end - enc_start # Calculate encryption time
        
        encrypted_size = len(serialized_vector)
        metrics["encryption_time"] = encryption_time # Log encryption time
        metrics["encrypted_size_bytes"] = encrypted_size
        metrics["expansion_factor"] = encrypted_size / plaintext_size if plaintext_size > 0 else 0
        
        payload = np.frombuffer(serialized_vector, dtype=np.uint8)

        # Debug print to confirm encryption time and sizes on client side
        print(f"[Client {partition_id}] Enc. Time: {encryption_time:.6f}s, Enc. Size: {encrypted_size} bytes, Plaintext Size: {plaintext_size} bytes, Expansion Factor: {metrics['expansion_factor']:.2f}")

        return [payload], len(self.trainloader.dataset), metrics

def client_fn_bfv(partition_id: int, run_config: dict, context_bytes: bytes):
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
    
    return BFVFlowerClient(net, trainloader, client_run_config, context).to_client()
