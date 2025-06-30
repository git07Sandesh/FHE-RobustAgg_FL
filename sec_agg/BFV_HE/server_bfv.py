# sec_agg/server_bfv.py

import wandb
import torch
import time
import numpy as np
from typing import List, Tuple, Optional, Dict # This was already here, which is good

import tenseal as ts
from flwr.server.strategy import FedAvg, Strategy
# FIXED: Import the necessary types from flwr
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from sec_agg.Plaintext.task import Net, SmallNet, get_central_testloader, get_weights, set_weights, test, unflatten_weights, flatten_weights
from .task_bfv import get_bfv_context, decode

# ==============================================================================
# BFV Strategy Implementations
# ==============================================================================

class BFVFedAvg(FedAvg):
    def __init__(self, fhe_context: ts.Context, sample_model: torch.nn.Module, precision_bits: int, **kwargs):
        self.fhe_context = fhe_context
        self.sample_model = sample_model
        self.precision_bits = precision_bits
        super().__init__(**kwargs)

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[any],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}
        
        # We need the plaintext number of examples for the final division
        num_examples_list = [res.num_examples for _, res in results]
        total_examples = sum(num_examples_list)
        if total_examples == 0: return None, {}

        # 1. Decrypt each client update individually to get its encoded integer vector
        # This is the "leaky" part of the protocol. The server learns the encoded updates.
        client_updates_encoded = [np.array(ts.bfv_vector_from(self.fhe_context, parameters_to_ndarrays(res.parameters)[0].tobytes()).decrypt()) for _, res in results]
        
        # 2. Perform the entire weighted average in PLAINTEXT on the encoded integers
        weighted_encoded_updates = [update * n_ex for update, n_ex in zip(client_updates_encoded, num_examples_list)]
        summed_weighted_updates = np.sum(np.array(weighted_encoded_updates), axis=0)
        averaged_encoded_vector = np.round(summed_weighted_updates / total_examples).astype(np.int64)

        # 3. Decode the final averaged integer vector back to floats
        decoded_flat_weights = decode(averaged_encoded_vector.tolist(), self.precision_bits)
        
        new_global_weights = unflatten_weights(decoded_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {}

class BFVMultiKrum(FedAvg):
    def __init__(self, num_malicious_clients: int, num_clients_to_keep: int, fhe_context: ts.Context, sample_model: torch.nn.Module, precision_bits: int, **kwargs):
        self.num_malicious_clients = num_malicious_clients
        self.num_clients_to_keep = num_clients_to_keep
        self.fhe_context = fhe_context
        self.sample_model = sample_model
        self.precision_bits = precision_bits
        super().__init__(**kwargs)

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[any]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print("BFVMultiKrum: Using leaky protocol (decrypt-all)")
        
        # 1. Decrypt all client updates to get the encoded integer vectors
        client_updates_encoded = [np.array(ts.bfv_vector_from(self.fhe_context, parameters_to_ndarrays(res.parameters)[0].tobytes()).decrypt()) for _, res in results]
        
        # 2. Compute plaintext scores on the ENCODED vectors
        n_clients = len(client_updates_encoded)
        scores = []
        for i in range(n_clients):
            dist_sum = 0
            for j in range(n_clients):
                if i == j: continue
                dist_sum += np.linalg.norm(client_updates_encoded[i] - client_updates_encoded[j]) ** 2
            scores.append(dist_sum)
            
        # 3. Select clients
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1])
        indices_to_keep = [idx for idx, _ in indexed_scores[:self.num_clients_to_keep]]
        print(f"BFVMultiKrum selected clients: {indices_to_keep}")
        
        # 4. Average the ENCODED updates of the selected clients in PLAINTEXT
        selected_updates_encoded = [client_updates_encoded[i] for i in indices_to_keep]
        averaged_encoded_vector = np.mean(np.array(selected_updates_encoded), axis=0).astype(np.int64)

        # 5. Decode the final averaged integer vector back to floats
        decoded_flat_weights = decode(averaged_encoded_vector.tolist(), self.precision_bits)
        
        new_global_weights = unflatten_weights(decoded_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {}
    # ==============================================================================
# The main builder function for ALL BFV strategies
# ==============================================================================
def build_strategy_bfv(run_config: dict) -> Tuple[Strategy, ts.Context]:
    strategy_name = run_config.get("strategy")
    
    wandb.init(project="fl-fhe-bfv-benchmark", name=run_config.get("run_name"), reinit=True, settings=wandb.Settings(start_method="thread"))
    wandb.config.update(run_config)

    fhe_context = get_bfv_context()
    
    if strategy_name == "BFVMultiKrum":
        model_to_use = SmallNet()
    else:
        model_to_use = Net()
    
    initial_parameters = ndarrays_to_parameters(get_weights(model_to_use))
    
    testloader = get_central_testloader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def evaluate(server_round: int, parameters: List[np.ndarray], _) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        set_weights(model_to_use, parameters)
        loss, acc = test(model_to_use, testloader, device)
        wandb.log({"round": server_round, "server_loss": loss, "server_accuracy": acc})
        return loss, {"accuracy": acc}

    def fit_config(server_round: int):
        return {"local_epochs": run_config.get("local_epochs", 1)}

    common_args = {
        "fhe_context": fhe_context, "sample_model": model_to_use,
        "fraction_fit": 1.0, "fraction_evaluate": 0.0,
        "min_available_clients": run_config["num_partitions"],
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate, "on_fit_config_fn": fit_config,
        "precision_bits": 16
    }

    if strategy_name == "BFVMultiKrum":
        strategy = BFVMultiKrum(num_malicious_clients=run_config.get("num_malicious", 0), num_clients_to_keep=run_config["num_partitions"] - run_config.get("num_malicious", 0), **common_args)
    else: # BFVFedAvg
        strategy = BFVFedAvg(**common_args)

    return strategy, fhe_context