# sec_agg/server_bfv.py

import wandb
import torch
import time
import numpy as np
from typing import List, Tuple, Optional, Dict

import tenseal as ts
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
        
        num_examples_list = [res.num_examples for _, res in results]
        total_examples = sum(num_examples_list)
        if total_examples == 0: return None, {}

        # "Leaky" protocol: Decrypt all updates first
        client_updates_encoded = [np.array(ts.bfv_vector_from(self.fhe_context, parameters_to_ndarrays(res.parameters)[0].tobytes()).decrypt()) for _, res in results]
        
        # Perform weighted average on the plaintext encoded integers
        weighted_encoded_updates = [update * n_ex for update, n_ex in zip(client_updates_encoded, num_examples_list)]
        summed_weighted_updates = np.sum(np.array(weighted_encoded_updates), axis=0)
        averaged_encoded_vector = np.round(summed_weighted_updates / total_examples).astype(np.int64)

        # Decode the final result
        decoded_flat_weights = decode(averaged_encoded_vector.tolist(), self.precision_bits)
        
        new_global_weights = unflatten_weights(decoded_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {}
class BFVKrum(FedAvg):
    def __init__(self, num_malicious_clients: int, fhe_context: ts.Context, sample_model: torch.nn.Module, precision_bits: int, **kwargs):
        self.num_malicious_clients = num_malicious_clients
        self.fhe_context = fhe_context
        self.sample_model = sample_model
        self.precision_bits = precision_bits
        super().__init__(**kwargs)

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[any]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}
        
        print("BFVKrum: Using leaky protocol (decrypt-all-to-compute-scores)")
        
        client_updates_encoded = [np.array(ts.bfv_vector_from(self.fhe_context, parameters_to_ndarrays(res.parameters)[0].tobytes()).decrypt()) for _, res in results]
        
        n_clients = len(client_updates_encoded)
        scores = []
        for i in range(n_clients):
            dist_sum = sum(np.linalg.norm(client_updates_encoded[i] - client_updates_encoded[j]) ** 2 for j in range(n_clients) if i != j)
            scores.append(dist_sum)
            
        # Find the client with minimum score (most similar to others)
        best_client_idx = np.argmin(scores)
        print(f"BFVKrum selected client with index: {best_client_idx}")
        
        # Use only the selected client's encoded update
        selected_update_encoded = client_updates_encoded[best_client_idx]
        
        decoded_flat_weights = decode(selected_update_encoded.tolist(), self.precision_bits)
        
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
        print("BFVMultiKrum: Using leaky protocol (decrypt-all-to-compute-scores)")
        
        client_updates_encoded = [np.array(ts.bfv_vector_from(self.fhe_context, parameters_to_ndarrays(res.parameters)[0].tobytes()).decrypt()) for _, res in results]
        
        n_clients = len(client_updates_encoded)
        scores = []
        for i in range(n_clients):
            dist_sum = sum(np.linalg.norm(client_updates_encoded[i] - client_updates_encoded[j]) ** 2 for j in range(n_clients) if i != j)
            scores.append(dist_sum)
            
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1])
        indices_to_keep = [idx for idx, _ in indexed_scores[:self.num_clients_to_keep]]
        print(f"BFVMultiKrum selected clients: {indices_to_keep}")
        
        selected_updates_encoded = [client_updates_encoded[i] for i in indices_to_keep]
        averaged_encoded_vector = np.mean(np.array(selected_updates_encoded), axis=0).astype(np.int64)

        decoded_flat_weights = decode(averaged_encoded_vector.tolist(), self.precision_bits)
        
        new_global_weights = unflatten_weights(decoded_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {}

class BFVTrimmedMean(FedAvg):
    def __init__(self, num_malicious_clients: int, fhe_context: ts.Context, sample_model: torch.nn.Module, precision_bits: int, **kwargs):
        super().__init__(**kwargs)
        self.b = num_malicious_clients
        self.fhe_context = fhe_context
        self.sample_model = sample_model
        self.precision_bits = precision_bits

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[any]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}

        print("BFVTrimmedMean: Using leaky protocol (decrypt-all-to-compute-norms)")

        client_updates_encoded = [np.array(ts.bfv_vector_from(self.fhe_context, parameters_to_ndarrays(res.parameters)[0].tobytes()).decrypt()) for _, res in results]
        
        norms = [np.linalg.norm(update) for update in client_updates_encoded]
        
        indexed_norms = sorted(enumerate(norms), key=lambda x: x[1])
        
        # --- FIXED: Simplified and Corrected Trimming Logic ---
        
        num_to_trim = self.b
        num_results = len(results)

        # Only trim if we have enough clients AND we are in an attack scenario
        if num_results > 2 * num_to_trim and num_to_trim > 0:
            untrimmed_indices = [i for i, _ in indexed_norms]
            indices_to_keep = untrimmed_indices[num_to_trim : -num_to_trim]
            print(f"BFVTrimmedMean trimmed {2 * num_to_trim} clients, keeping {len(indices_to_keep)}.")
        else:
            # In all other cases (benign run, or not enough clients), keep everyone
            indices_to_keep = list(range(num_results))
            print(f"BFVTrimmedMean: Not enough clients to trim, or benign run. Keeping all {len(indices_to_keep)} clients.")
        
        # --- End of fixed logic ---

        if not indices_to_keep:
            print("WARNING: BFVTrimmedMean ended up with no clients to keep. This should not happen.")
            return None, {}

        selected_updates_encoded = [client_updates_encoded[i] for i in indices_to_keep]
        averaged_encoded_vector = np.mean(np.array(selected_updates_encoded), axis=0).astype(np.int64)
        
        decoded_flat_weights = decode(averaged_encoded_vector.tolist(), self.precision_bits)
        
        new_global_weights = unflatten_weights(decoded_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {}
    
class BFVBulyan(FedAvg):
    # This __init__ was already correct from the last step
    def __init__(self, num_malicious_clients: int, fhe_context: ts.Context, sample_model: torch.nn.Module, precision_bits: int,
                 bulyan_selection_size: Optional[int] = None, trimmed_mean_beta: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.f = num_malicious_clients
        self.fhe_context = fhe_context
        self.sample_model = sample_model
        self.precision_bits = precision_bits
        self.bulyan_selection_size = bulyan_selection_size
        self.trimmed_mean_beta = trimmed_mean_beta if trimmed_mean_beta is not None else num_malicious_clients

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[any]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: 
            return None, {}
        
        n = len(results)
        f = self.f
        
        theta = self.bulyan_selection_size if self.bulyan_selection_size is not None else n - 2*f
        beta = self.trimmed_mean_beta
        
        print(f"BFVBulyan parameters: theta={theta}, beta={beta}, n_clients={n}, f={f}")

        if n < theta:
             print(f"WARNING: Not enough clients ({n}) to select {theta}. Falling back to simple average.")
             return self._fallback_average(results)
        
        print(f"BFVBulyan: Using leaky protocol (decrypt-all-to-compute-selection)")
        
        client_updates_encoded = []
        for _, res in results:
            encrypted_params = parameters_to_ndarrays(res.parameters)[0].tobytes()
            decrypted_vector = ts.bfv_vector_from(self.fhe_context, encrypted_params).decrypt()
            client_updates_encoded.append(np.array(decrypted_vector))
        
        # This call is correct
        selected_updates = self._multi_krum_selection(client_updates_encoded, f, theta)
        
        if not selected_updates:
            print("WARNING: Multi-Krum selection returned no clients. Falling back to simple average.")
            return self._fallback_average(results)
        
        print(f"BFVBulyan: Multi-Krum selected {len(selected_updates)} clients")
        
        # This call is correct
        aggregated_update = self._coordinate_wise_trimmed_mean(selected_updates, beta)
        
        decoded_flat_weights = decode(aggregated_update.tolist(), self.precision_bits)
        new_global_weights = unflatten_weights(decoded_flat_weights, self.sample_model)
        
        return ndarrays_to_parameters(new_global_weights), {}
    
    # ***** FIX #1: THE SIGNATURE FOR THIS FUNCTION *****
    def _multi_krum_selection(self, client_updates_encoded: List[np.ndarray], f: int, theta: int) -> List[np.ndarray]:
        """
        Multi-Krum selection: Select `theta` clients with smallest Krum scores.
        """
        n = len(client_updates_encoded)
        m = n - f - 2
        
        if m <= 0: return []
        
        scores = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(client_updates_encoded[i] - client_updates_encoded[j]) ** 2
                    distances.append(dist)
            
            distances.sort()
            krum_score = sum(distances[:m])
            scores.append((i, krum_score))
        
        scores.sort(key=lambda x: x[1])
        # Use the passed-in 'theta' parameter
        selected_indices = [idx for idx, _ in scores[:theta]]
        
        return [client_updates_encoded[i] for i in selected_indices]

    # ***** FIX #2: THE SIGNATURE FOR THIS FUNCTION *****
    def _coordinate_wise_trimmed_mean(self, selected_updates: List[np.ndarray], beta: int) -> np.ndarray:
        """
        Coordinate-wise trimmed mean: For each coordinate, remove `beta` smallest and `beta` largest values.
        """
        if not selected_updates:
            return np.array([])
        
        updates_matrix = np.array(selected_updates)
        num_clients, num_params = updates_matrix.shape
        
        if num_clients < 2 * beta + 1:
            print(f"WARNING: Not enough clients ({num_clients}) for trimmed mean with beta={beta}. Using simple mean.")
            return np.mean(updates_matrix, axis=0).astype(np.int64)
        
        # Use the passed-in 'beta' parameter
        print(f"BFVBulyan Phase 2: Applying coordinate-wise trimmed mean to {num_clients} updates with beta={beta}.")
        aggregated = np.zeros(num_params, dtype=np.int64)
        
        for coord in range(num_params):
            coord_values = updates_matrix[:, coord]
            sorted_values = np.sort(coord_values)
            
            # Use the passed-in 'beta' parameter
            if len(sorted_values) > 2 * beta:
                trimmed_values = sorted_values[beta:-beta] if beta > 0 else sorted_values
            else:
                trimmed_values = sorted_values
            
            aggregated[coord] = np.round(np.mean(trimmed_values)).astype(np.int64)
        
        return aggregated
    
    def _fallback_average(self, results: List[Tuple[ClientProxy, FitRes]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # This function was already correct
        print("BFVBulyan: Falling back to weighted average")
        num_examples_list = [res.num_examples for _, res in results]
        total_examples = sum(num_examples_list)
        if total_examples == 0: 
            return None, {}
        client_updates_encoded = []
        for _, res in results:
            encrypted_params = parameters_to_ndarrays(res.parameters)[0].tobytes()
            decrypted_vector = ts.bfv_vector_from(self.fhe_context, encrypted_params).decrypt()
            client_updates_encoded.append(np.array(decrypted_vector))
        weighted_encoded_updates = [update * n_ex for update, n_ex in zip(client_updates_encoded, num_examples_list)]
        summed_weighted_updates = np.sum(np.array(weighted_encoded_updates), axis=0)
        averaged_encoded_vector = np.round(summed_weighted_updates / total_examples).astype(np.int64)
        decoded_flat_weights = decode(averaged_encoded_vector.tolist(), self.precision_bits)
        new_global_weights = unflatten_weights(decoded_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {}
   
# ==============================================================================
# The main builder function for ALL BFV strategies
# ==============================================================================
def build_strategy_bfv(run_config: dict) -> Tuple[Strategy, ts.Context]:
    strategy_name = run_config.get("strategy")
    num_malicious = run_config.get("num_malicious", 0)
    wandb.init(project="fl-fhe-bfv-benchmark", name=run_config.get("run_name"), reinit=True, settings=wandb.Settings(start_method="thread"))
    wandb.config.update(run_config)

    fhe_context = get_bfv_context()

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
        "fhe_context": fhe_context, "sample_model": model_to_use,
        "fraction_fit": 1.0, "fraction_evaluate": 0.0,
        "min_available_clients": run_config["num_partitions"],
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate, "on_fit_config_fn": fit_config,
        "precision_bits": 16
    }
    if strategy_name == "BFVKrum":
        strategy = BFVKrum(num_malicious_clients=run_config.get("num_malicious", 0), **common_args)
    elif strategy_name == "BFVMultiKrum":
        strategy = BFVMultiKrum(num_malicious_clients=run_config.get("num_malicious", 0), num_clients_to_keep=run_config["num_partitions"] - run_config.get("num_malicious", 0), **common_args)
    elif strategy_name == "BFVTrimmedMean":
        strategy = BFVTrimmedMean(num_malicious_clients=run_config.get("num_malicious", 0), **common_args)
    elif strategy_name == "BFVBulyan":
        # Get the Bulyan-specific parameters from the config
        selection_size = run_config.get("bulyan_selection_size")
        trimmed_mean_beta = run_config.get("trimmed_mean_beta")

        strategy = BFVBulyan(
            num_malicious_clients=num_malicious,
            # --- PASS THE PARAMETERS ---
            bulyan_selection_size=selection_size,
            trimmed_mean_beta=trimmed_mean_beta,
            **common_args
        )
    else: # BFVFedAvg
        strategy = BFVFedAvg(**common_args)

    return strategy, fhe_context