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
from sec_agg.Plaintext.task import Net, get_central_testloader, get_weights, set_weights, test, unflatten_weights, flatten_weights
from .task_paillier import PaillierContext, decode_weights, decrypt_vector

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
        
        # Paillier aggregation works with EncryptedNumber objects, not raw bytes.
        # We deserialize them from the payload.
        client_updates = [pickle.loads(parameters_to_ndarrays(res.parameters)[0].tobytes()) for _, res in results]
        num_examples_list = [res.num_examples for _, res in results]
        
        # Perform weighted average on encrypted numbers
        total_examples = sum(num_examples_list)
        if total_examples == 0: return None, {}
        
        # Summing encrypted vectors is direct homomorphic addition
        weighted_sum = sum(vec * num_examples for vec, num_examples in zip(client_updates, num_examples_list))
        
        # Division is multiplication by the plaintext inverse
        aggregated_encoded_vector = weighted_sum * (1 / total_examples)

        # Decrypt and decode
        decrypted_vector = decrypt_vector(self.p_context.private_key, aggregated_encoded_vector)
        decoded_flat_weights = decode_weights(decrypted_vector, self.p_context.precision)
        
        new_global_weights = unflatten_weights(decoded_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {}

class PaillierTrimmedMean(FedAvg):
    def __init__(self, num_malicious_clients: int, paillier_context: PaillierContext, sample_model: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.b = num_malicious_clients
        self.p_context = paillier_context
        self.sample_model = sample_model

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}

        norms = [res.metrics.get("l2_norm", 0.0) for _, res in results]
        indexed_norms = sorted(enumerate(norms), key=lambda x: x[1])
        
        num_to_trim = self.b
        if len(indexed_norms) <= 2 * num_to_trim:
            indices_to_keep = list(range(len(indexed_norms)))
        else:
            indices_to_keep = [i for i, _ in indexed_norms][num_to_trim:-num_to_trim]

        print(f"TrimmedMean kept {len(indices_to_keep)} clients: {indices_to_keep}")
        
        client_updates = [pickle.loads(parameters_to_ndarrays(results[i][1].parameters)[0].tobytes()) for i in indices_to_keep]

        if not client_updates: return None, {}
        
        aggregated_vector = sum(client_updates) * (1 / len(client_updates))
        
        decrypted_vector = decrypt_vector(self.p_context.private_key, aggregated_vector)
        decoded_flat_weights = decode_weights(decrypted_vector, self.p_context.precision)
        
        new_global_weights = unflatten_weights(decoded_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {}

class PaillierKrum(FedAvg):
    """
    Krum/Multi-Krum implementation using a 'decrypt-to-compute' leaky protocol.
    The server decrypts all updates to compute distances for client selection,
    then aggregates the original encrypted updates of the selected clients.
    """
    def __init__(
        self,
        num_malicious_clients: int,
        num_clients_to_select: int,
        paillier_context: PaillierContext,
        sample_model: torch.nn.Module,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.b = num_malicious_clients
        self.k = num_clients_to_select
        self.p_context = paillier_context
        self.sample_model = sample_model
        print(f"Initialized PaillierKrum: b={self.b}, k={self.k}")

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}

        # 1. Deserialize all client updates
        encrypted_updates = [pickle.loads(parameters_to_ndarrays(res.parameters)[0].tobytes()) for _, res in results]
        num_clients = len(encrypted_updates)

        # 2. (THE LEAK) Decrypt all updates to compute distances in plaintext
        print("PaillierKrum: Decrypting updates for distance calculation (privacy leak).")
        decryption_start = time.time()
        decrypted_updates = [
            decode_weights(
                decrypt_vector(self.p_context.private_key, enc_up), 
                self.p_context.precision
            ) for enc_up in encrypted_updates
        ]
        decryption_end = time.time()
        print(f"PaillierKrum: Decryption took {decryption_end - decryption_start:.2f}s")
        
        # 3. Compute pairwise squared Euclidean distances
        distances = np.array([
            [np.linalg.norm(u1 - u2) ** 2 for u2 in decrypted_updates]
            for u1 in decrypted_updates
        ])
        
        # 4. Compute Krum scores
        # Score for client i is the sum of distances to its n-b-2 closest neighbors
        num_closest = num_clients - self.b - 2
        if num_closest < 1:
            print(f"Warning: Not enough clients ({num_clients}) to apply Krum with b={self.b}. Selecting all.")
            indices_to_keep = list(range(num_clients))
        else:
            scores = []
            for i in range(num_clients):
                # Sort distances to other clients and sum the smallest `num_closest`
                sorted_dists = np.sort(distances[i])
                # Exclude self-distance (which is 0) by starting from index 1
                scores.append(np.sum(sorted_dists[1:num_closest + 1]))
            
            # 5. Select the k clients with the lowest scores
            sorted_indices = np.argsort(scores)
            indices_to_keep = sorted_indices[:self.k].tolist()

        print(f"Krum kept {len(indices_to_keep)} clients: {indices_to_keep}")
        
        # 6. Aggregate the *original encrypted* updates from the selected clients
        selected_encrypted_updates = [encrypted_updates[i] for i in indices_to_keep]
        
        if not selected_encrypted_updates: return None, {}
        
        # Simple averaging of the selected updates
        aggregated_vector = sum(selected_encrypted_updates) * (1 / len(selected_encrypted_updates))
        
        # 7. Decrypt final result and convert back to model weights
        decrypted_vector = decrypt_vector(self.p_context.private_key, aggregated_vector)
        decoded_flat_weights = decode_weights(decrypted_vector, self.p_context.precision)
        
        new_global_weights = unflatten_weights(decoded_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {}

# ==============================================================================
# The main builder function for Paillier strategies
# ==============================================================================
def build_strategy_paillier(run_config: dict) -> Tuple[Strategy, PaillierContext]:
    strategy_name = run_config.get("strategy")
    num_malicious = run_config.get("num_malicious", 0)
    num_partitions = run_config["num_partitions"]

    wandb.init(project="fl-fhe-paillier-benchmark", name=run_config.get("run_name"), reinit=True, settings=wandb.Settings(start_method="thread"))
    wandb.config.update(run_config)

    p_context = PaillierContext()
    
    model_to_use = Net()
    initial_parameters = ndarrays_to_parameters(get_weights(model_to_use))
    
    testloader = get_central_testloader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round: int, parameters: List[np.ndarray], _):
        # The parameters are already decrypted NDArrays at this stage
        set_weights(model_to_use, parameters)
        loss, acc = test(model_to_use, testloader, device)
        wandb.log({"round": server_round, "server_loss": loss, "server_accuracy": acc})
        return loss, {"accuracy": acc}

    def fit_config(server_round: int):
        return {"local_epochs": run_config.get("local_epochs", 1)}
    
    common_args = {
        "paillier_context": p_context,
        "sample_model": model_to_use,
        "fraction_fit": 1.0, "fraction_evaluate": 1.0, # Evaluate on server-side
        "min_available_clients": num_partitions,
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate, "on_fit_config_fn": fit_config,
    }
    
    if strategy_name == "PaillierTrimmedMean":
        strategy = PaillierTrimmedMean(num_malicious_clients=num_malicious, **common_args)
    elif strategy_name == "PaillierKrum":
        strategy = PaillierKrum(
            num_malicious_clients=num_malicious,
            num_clients_to_select=1,
            **common_args
        )
    elif strategy_name == "PaillierMultiKrum":
        # Multi-Krum selects m = n - b - 2 clients
        num_to_select = num_partitions - num_malicious - 2
        if num_to_select < 1:
            print(f"Warning: MultiKrum wants to select {num_to_select} clients. Defaulting to 1.")
            num_to_select = 1
        strategy = PaillierKrum(
            num_malicious_clients=num_malicious,
            num_clients_to_select=num_to_select,
            **common_args
        )
    else: # Default to PaillierFedAvg
        strategy = PaillierFedAvg(**common_args)

    return strategy, p_context