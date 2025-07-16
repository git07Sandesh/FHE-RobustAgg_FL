# sec_agg/client_paillier.py

import torch
import numpy as np
import time
import pickle
import sys # Import sys for sys.getsizeof
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
        
        train_start_time = time.perf_counter() # Use perf_counter
        trained_weights = train(self.net, self.trainloader, epochs=config["local_epochs"], device=DEVICE)
        train_end_time = time.perf_counter()
        training_time = train_end_time - train_start_time
        
        partition_id = self.run_config.get("partition_id", -1)
        is_byzantine = partition_id < self.run_config.get("num_malicious", 0) # Check if client is malicious

        if is_byzantine:
            attack_type = self.run_config.get("attack_type", "gaussian_noise")
            sigma = self.run_config.get("attack_sigma", 0.5)
            print(f"[Client {partition_id}] Byzantine client applying '{attack_type}' attack with sigma={sigma}")
            # Ensure the attack is applied correctly; -w might not be ideal for all attacks.
            # Assuming for now, the intent is just a simple sign-flipping.
            trained_weights = [w + np.random.normal(0, sigma, w.shape).astype(np.float32) for w in trained_weights]


        metrics = {
            "training_time": training_time, # Add training time
        }
        
        strategy_name = self.run_config.get("strategy")
        # For TrimmedMean/Krum, send the plaintext L2 norm
        if strategy_name in ["PaillierTrimmedMean", "PaillierKrum"]: # Added Krum
            flat_weights = flatten_weights(trained_weights)
            l2_norm = np.linalg.norm(flat_weights)
            metrics["l2_norm"] = float(l2_norm) # Ensure it's a float
            
        # For all schemes, log plaintext payload size
        plaintext_payload_bytes = pickle.dumps(trained_weights)
        plaintext_size_bytes = sys.getsizeof(plaintext_payload_bytes)
        metrics["plaintext_size_bytes"] = plaintext_size_bytes
            
        # Encrypt the list of weight tensors
        enc_start = time.perf_counter() # Measure encryption time
        encrypted_update = encrypt_weights(self.public_key, trained_weights)
        
        # Serialize the list of EncryptedNumber arrays using pickle
        payload_bytes = pickle.dumps(encrypted_update)
        enc_end = time.perf_counter()
        encryption_time = enc_end - enc_start # Calculate encryption time
        
        encrypted_size_bytes = len(payload_bytes) # Use len() on bytes object
        
        metrics["encryption_time"] = encryption_time # Log encryption time
        metrics["encrypted_size_bytes"] = encrypted_size_bytes
        metrics["expansion_factor"] = encrypted_size_bytes / plaintext_size_bytes if plaintext_size_bytes > 0 else 0
        
        payload = np.frombuffer(payload_bytes, dtype=np.uint8) # Convert bytes to uint8 numpy array for Flower
        
        # Debug print
        print(f"[Client {partition_id}] Enc. Time: {encryption_time:.6f}s, Enc. Size: {encrypted_size_bytes} bytes, Plaintext Size: {plaintext_size_bytes} bytes, Expansion Factor: {metrics['expansion_factor']:.2f}")

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
        paillier_context.public_key # Pass only the public key
    ).to_client()
