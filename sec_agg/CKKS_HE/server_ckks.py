# sec_agg/server_ckks.py

import wandb
import torch
import time
import tenseal as ts
import numpy as np
from typing import List, Tuple, Union, Optional, Dict

from flwr.server.strategy import FedAvg, Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from sec_agg.Plaintext.task import Net, SmallNet, get_central_testloader, get_weights, set_weights, test, unflatten_weights, flatten_weights
from .task_ckks import get_ckks_context

# ==============================================================================
# CKKS Strategy Implementations
# ==============================================================================

class CKKSFedAvg(FedAvg):
    def __init__(self, fhe_context: ts.Context, sample_model: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.fhe_context = fhe_context
        self.sample_model = sample_model

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}
        
        num_examples_total = 0
        weighted_vectors = []
        for _, fit_res in results:
            uint8_array = parameters_to_ndarrays(fit_res.parameters)[0]
            encrypted_vector = ts.ckks_vector_from(self.fhe_context, uint8_array.tobytes())
            num_examples = fit_res.num_examples
            encrypted_vector *= num_examples
            weighted_vectors.append(encrypted_vector)
            num_examples_total += num_examples
            
        if not weighted_vectors or num_examples_total == 0: return None, {}
        
        aggregated_vector = sum(weighted_vectors) * (1 / num_examples_total)
        
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())
        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {}

class CKKSMultiKrum(FedAvg):
    def __init__(self, num_malicious_clients: int, num_clients_to_keep: int, fhe_context: ts.Context, sample_model: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious_clients = num_malicious_clients
        self.num_clients_to_keep = num_clients_to_keep
        self.fhe_context = fhe_context
        self.sample_model = sample_model

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}

        client_vectors = [ts.ckks_vector_from(self.fhe_context, parameters_to_ndarrays(res.parameters)[0].tobytes()) for _, res in results]
        n_clients = len(client_vectors)

        print("Homomorphically computing pairwise squared distances...")
        scores = []
        for i in range(n_clients):
            distances = []
            m = n_clients - self.num_malicious_clients - 2
            # The original Krum paper suggests summing the distances to the n-f-2 closest neighbors.
            # A common simplification is to sum all distances, which we do here.
            for j in range(n_clients):
                if i == j: continue
                distance_vec = client_vectors[i] - client_vectors[j]
                squared_distance = distance_vec.dot(distance_vec) 
                distances.append(squared_distance)
            scores.append(sum(distances))
        
        print("Decrypting Krum scores...")
        decrypted_scores = [score.decrypt()[0] for score in scores]
        
        indexed_scores = sorted(enumerate(decrypted_scores), key=lambda x: x[1])
        indices_to_keep = [idx for idx, _ in indexed_scores[:self.num_clients_to_keep]]
        print(f"Multi-Krum selected clients with indices: {indices_to_keep}")
        
        selected_updates = [client_vectors[i] for i in indices_to_keep]
        aggregated_vector = sum(selected_updates) * (1 / len(selected_updates))
        
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())
        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        
        return ndarrays_to_parameters(new_global_weights), {}

class CKKSTrimmedMean(FedAvg):
    def __init__(self, num_malicious_clients: int, fhe_context: ts.Context, sample_model: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.b = num_malicious_clients
        self.fhe_context = fhe_context
        self.sample_model = sample_model

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}

        # "Leaky" protocol: Get plaintext L2 norms from client metrics
        norms = [res.metrics.get("l2_norm", 0.0) for _, res in results]
        
        indexed_norms = sorted(enumerate(norms), key=lambda x: x[1])
        
        num_to_trim = self.b
        if len(indexed_norms) <= 2 * num_to_trim:
            indices_to_keep = list(range(len(indexed_norms)))
        else:
            untrimmed_indices = [i for i, _ in indexed_norms]
            indices_to_keep = untrimmed_indices[num_to_trim : -num_to_trim]

        print(f"TrimmedMean kept {len(indices_to_keep)} clients: {indices_to_keep}")
        
        # Get encrypted vectors from the original results list using the kept indices
        client_vectors = [ts.ckks_vector_from(self.fhe_context, parameters_to_ndarrays(results[i][1].parameters)[0].tobytes()) for i in indices_to_keep]
        
        if not client_vectors: return None, {}

        aggregated_vector = sum(client_vectors) * (1 / len(client_vectors))
        
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())
        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        
        return ndarrays_to_parameters(new_global_weights), {}

# ==============================================================================
# The main builder function for ALL CKKS strategies
# ==============================================================================
def build_strategy_ckks(run_config: dict) -> Tuple[Strategy, bytes]:
    strategy_name = run_config.get("strategy")
    num_malicious = run_config.get("num_malicious", 0)

    wandb.init(project="fl-fhe-ckks-benchmark", name=run_config.get("run_name"), reinit=True, settings=wandb.Settings(start_method="thread"))
    wandb.config.update(run_config)

    fhe_context = get_ckks_context()
    context_bytes = fhe_context.serialize(save_secret_key=False)

    # MultiKrum requires the smaller model for FHE dot products.
    if strategy_name == "CKKSMultiKrum":
        model_to_use = SmallNet()
    else: # CKKSFedAvg and CKKSTrimmedMean can use the larger model
        model_to_use = Net()
    print(f"✅ Server using model: {model_to_use.__class__.__name__}")
    
    initial_parameters = ndarrays_to_parameters(get_weights(model_to_use))
    
    testloader = get_central_testloader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round: int, parameters: List[np.ndarray], _):
        set_weights(model_to_use, parameters)
        loss, acc = test(model_to_use, testloader, device)
        wandb.log({"round": server_round, "server_loss": loss, "server_accuracy": acc})
        return loss, {"accuracy": acc}

    def fit_config(server_round: int):
        return {"local_epochs": run_config.get("local_epochs", 1), "fhe_context_bytes": context_bytes}
    
    common_args = {
        "fhe_context": fhe_context,
        "sample_model": model_to_use,
        "fraction_fit": 1.0,
        "fraction_evaluate": 0.0,
        "min_available_clients": run_config["num_partitions"],
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate,
        "on_fit_config_fn": fit_config,
    }

    if strategy_name == "CKKSMultiKrum":
        strategy = CKKSMultiKrum(num_malicious_clients=num_malicious, num_clients_to_keep=run_config["num_partitions"] - num_malicious, **common_args)
    elif strategy_name == "CKKSTrimmedMean":
        strategy = CKKSTrimmedMean(num_malicious_clients=num_malicious, **common_args)
    else: # Default to CKKSFedAvg
        strategy = CKKSFedAvg(**common_args)

    return strategy, context_bytes