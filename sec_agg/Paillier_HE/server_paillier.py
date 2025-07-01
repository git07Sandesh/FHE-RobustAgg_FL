# sec_agg/server_paillier.py

import wandb
import torch
import time
import pickle
import numpy as np
from typing import List, Tuple, Optional, Dict

from flwr.server.strategy import FedAvg, Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from sec_agg.Plaintext.task import Net, SmallNet, get_central_testloader, get_weights, set_weights, test
from .task_paillier import PaillierContext, decrypt_weights

# ==============================================================================
# Paillier Strategy Implementations
# ==============================================================================

class PaillierFedAvg(FedAvg):
    def __init__(self, paillier_context: PaillierContext, sample_model: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.p_context = paillier_context
        self.sample_model = sample_model

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}
        
        client_updates = [pickle.loads(parameters_to_ndarrays(res.parameters)[0].tobytes()) for _, res in results]
        num_examples_list = [res.num_examples for _, res in results]
        
        total_examples = sum(num_examples_list)
        if total_examples == 0: return None, {}
        
        weighted_updates = []
        for update, num_examples in zip(client_updates, num_examples_list):
            weighted_layer = [layer * num_examples for layer in update]
            weighted_updates.append(weighted_layer)
        
        summed_updates = [np.sum(layers, axis=0) for layers in zip(*weighted_updates)]
        
        aggregated_encrypted_vector = [layer * (1.0 / total_examples) for layer in summed_updates]

        decrypted_weights = decrypt_weights(self.p_context.private_key, aggregated_encrypted_vector)
        return ndarrays_to_parameters(decrypted_weights), {}

class PaillierTrimmedMean(FedAvg):
    def __init__(self, num_malicious_clients: int, paillier_context: PaillierContext, sample_model: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.num_to_trim = num_malicious_clients
        self.p_context = paillier_context
        self.sample_model = sample_model

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}

        # "Leaky" protocol: Get plaintext L2 norms from client metrics
        norms = [res.metrics.get("l2_norm", float('inf')) for _, res in results]
        
        # Associate norms with their original index in the `results` list
        indexed_norms = sorted(enumerate(norms), key=lambda x: x[1])
        
        # --- DEFINITIVE CORRECTED LOGIC ---
        # Determine how many clients to keep
        num_to_keep = len(results) - self.num_to_trim
        
        # Get the indices of the clients with the smallest norms
        indices_to_keep = [i for i, norm in indexed_norms[:num_to_keep]]

        print(f"TrimmedMean trimmed {self.num_to_trim} clients, keeping {len(indices_to_keep)}.")
        
        # If no clients are left after trimming, stop.
        if not indices_to_keep:
            return None, {}
        
        # Get the encrypted updates from the selected clients using their original indices
        client_updates = [pickle.loads(parameters_to_ndarrays(results[i][1].parameters)[0].tobytes()) for i in indices_to_keep]

        # Homomorphically average the selected updates
        summed_updates = [np.sum(layers, axis=0) for layers in zip(*client_updates)]
        aggregated_vector = [layer * (1.0 / len(client_updates)) for layer in summed_updates]
        
        decrypted_weights = decrypt_weights(self.p_context.private_key, aggregated_vector)
        return ndarrays_to_parameters(decrypted_weights), {}

class PaillierKrum(FedAvg):
    def __init__(self, num_malicious_clients: int, paillier_context: PaillierContext, sample_model: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious = num_malicious_clients
        self.p_context = paillier_context
        self.sample_model = sample_model

    def compute_pairwise_distances(self, updates: List[List[np.ndarray]]) -> np.ndarray:
        """
        Compute pairwise squared Euclidean distances between client updates.
        This requires decrypting the updates first (leaky but necessary for Krum).
        """
        n = len(updates)
        distances = np.zeros((n, n))
        
        # Decrypt all updates for distance computation
        decrypted_updates = []
        for update in updates:
            decrypted_update = decrypt_weights(self.p_context.private_key, update)
            # Flatten all layers into a single vector for distance computation
            flattened = np.concatenate([layer.flatten() for layer in decrypted_update])
            decrypted_updates.append(flattened)
        
        # Compute pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sum((decrypted_updates[i] - decrypted_updates[j]) ** 2)
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: 
            return None, {}
        
        n = len(results)
        if n <= self.num_malicious:
            print(f"Krum: Not enough clients ({n}) for num_malicious={self.num_malicious}")
            return None, {}
        
        # Extract encrypted client updates
        client_updates = [pickle.loads(parameters_to_ndarrays(res.parameters)[0].tobytes()) for _, res in results]
        
        # Compute pairwise distances (requires decryption - leaky protocol)
        distances = self.compute_pairwise_distances(client_updates)
        
        # For each client, compute the sum of distances to its n-f-2 closest neighbors
        # where f is the number of malicious clients
        num_closest = n - self.num_malicious - 2
        if num_closest <= 0:
            print(f"Krum: Invalid configuration - need at least {self.num_malicious + 3} clients for num_malicious={self.num_malicious}")
            return None, {}
        
        scores = []
        for i in range(n):
            # Get distances from client i to all other clients
            client_distances = distances[i, :]
            # Remove distance to self (which is 0)
            other_distances = np.concatenate([client_distances[:i], client_distances[i+1:]])
            # Sort and take the num_closest smallest distances
            closest_distances = np.partition(other_distances, num_closest)[:num_closest]
            # Sum of distances to closest neighbors
            score = np.sum(closest_distances)
            scores.append(score)
        
        # Select the client with the minimum score (closest to its neighbors)
        selected_client_idx = np.argmin(scores)
        
        print(f"Krum selected client {selected_client_idx} with score {scores[selected_client_idx]:.4f}")
        
        # Return the selected client's update (decrypt it first)
        selected_update = client_updates[selected_client_idx]
        decrypted_weights = decrypt_weights(self.p_context.private_key, selected_update)
        
        return ndarrays_to_parameters(decrypted_weights), {"krum_selected_client": selected_client_idx}


# ==============================================================================
# The main builder function for Paillier strategies
# ==============================================================================
def build_strategy_paillier(run_config: dict) -> Tuple[Strategy, PaillierContext]:
    strategy_name = run_config.get("strategy")
    num_malicious = run_config.get("num_malicious", 0)

    wandb.init(project="fl-paillier-benchmark", name=run_config.get("run_name"), reinit=True, settings=wandb.Settings(start_method="thread"))
    wandb.config.update(run_config)

    p_context = PaillierContext(key_size=512) # Using 512 for speed
    
    # Using SmallNet for all Paillier runs to ensure reasonable performance
    model_to_use = SmallNet()
    initial_parameters = ndarrays_to_parameters(get_weights(model_to_use))
    
    testloader = get_central_testloader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round: int, parameters: List[np.ndarray], _):
        set_weights(model_to_use, parameters)
        loss, acc = test(model_to_use, testloader, device)
        wandb.log({"round": server_round, "server_loss": loss, "server_accuracy": acc})
        return loss, {"accuracy": acc}

    def fit_config(server_round: int):
        return {"local_epochs": run_config.get("local_epochs", 1)}
    
    common_args = {
        "paillier_context": p_context,
        "sample_model": model_to_use,
        "fraction_fit": 1.0, "fraction_evaluate": 0.0,
        "min_available_clients": run_config["num_partitions"],
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate, "on_fit_config_fn": fit_config,
    }
    
    if strategy_name == "PaillierTrimmedMean":
        strategy = PaillierTrimmedMean(num_malicious_clients=num_malicious, **common_args)
    elif strategy_name == "PaillierKrum":
        strategy = PaillierKrum(num_malicious_clients=num_malicious, **common_args)
    else: # Default to PaillierFedAvg
        strategy = PaillierFedAvg(**common_args)

    return strategy, p_context