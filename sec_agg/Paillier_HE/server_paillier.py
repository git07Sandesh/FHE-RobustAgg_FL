# sec_agg/server_paillier.py

import wandb
import torch
import time
import pickle
import numpy as np
import os # Import os for artifact path
import json # Import json for artifact data
from typing import List, Tuple, Optional, Dict, Union # Added Union for failures

from flwr.server.strategy import FedAvg, Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from sec_agg.Plaintext.task import Net, SmallNet, get_central_testloader, get_weights, set_weights, test, unflatten_weights, flatten_weights, get_weights_size_bytes
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
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]], # Changed to Union
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}
        
        client_updates = [pickle.loads(parameters_to_ndarrays(res.parameters)[0].tobytes()) for _, res in results]
        num_examples_list = [res.num_examples for _, res in results]
        
        total_examples = sum(num_examples_list)
        if total_examples == 0: return None, {}
        
        weighted_updates = []
        for update, num_examples in zip(client_updates, num_examples_list):
            # Sum encrypted_number arrays element-wise for each layer
            weighted_layer = [layer_np_array * num_examples for layer_np_array in update]
            weighted_updates.append(weighted_layer)
        
        # Sum layers across all clients (element-wise aggregation of EncryptedNumber arrays)
        # Needs to convert list of lists of EncryptedNumbers to list of NumPy arrays of EncryptedNumbers
        # for np.sum to work correctly on the array-like structures
        summed_updates_raw = []
        for i in range(len(weighted_updates[0])): # Iterate through layers
            layer_sum = weighted_updates[0][i] # Start with first client's layer
            for j in range(1, len(weighted_updates)): # Add layers from subsequent clients
                layer_sum += weighted_updates[j][i]
            summed_updates_raw.append(layer_sum)

        aggregated_encrypted_vector = [layer_sum_array * (1.0 / total_examples) for layer_sum_array in summed_updates_raw]
        
        # --- Measure server-side decryption time ---
        dec_start = time.perf_counter()
        decrypted_weights = decrypt_weights(self.p_context.private_key, aggregated_encrypted_vector)
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start
        # -------------------------------------------

        # Flatten decrypted weights to match expected format for unflatten_weights (if needed)
        # Note: decrypt_weights already returns unflattened structure
        new_global_weights = decrypted_weights # Assumes decrypted_weights is already in list of np.ndarray format
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}

class PaillierTrimmedMean(FedAvg):
    def __init__(self, num_malicious_clients: int, paillier_context: PaillierContext, sample_model: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.num_to_trim = num_malicious_clients
        self.p_context = paillier_context
        self.sample_model = sample_model

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]], # Changed to Union
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}

        # "Leaky" protocol: Get plaintext L2 norms from client metrics
        norms = [res.metrics.get("l2_norm", float('inf')) for _, res in results]
        
        # Associate norms with their original index in the `results` list
        indexed_norms = sorted(enumerate(norms), key=lambda x: x[1])
        
        # --- DEFINITIVE CORRECTED LOGIC ---
        # Determine how many clients to keep
        # Trim `num_to_trim` smallest and `num_to_trim` largest
        num_to_trim = self.num_to_trim # Use the stored value directly
        num_clients_available = len(indexed_norms)
        
        if num_to_trim == 0:
            # If no malicious clients, no trimming should occur; keep all.
            indices_to_keep = list(range(num_clients_available))
            print(f"PaillierTrimmedMean: Benign scenario, keeping all {len(indices_to_keep)} clients.")
        elif num_clients_available <= 2 * num_to_trim:
            # If not enough clients to trim from both sides, or trimming would remove too many.
            # In such cases, standard Trimmed Mean might fail. Keep all as a safer default.
            indices_to_keep = list(range(num_clients_available))
            print(f"Warning: PaillierTrimmedMean intended to trim {num_to_trim} from each side but only {num_clients_available} clients available. Keeping all clients due to insufficient count.")
        else:
            # Standard trimming: remove 'num_to_trim' from each end
            untrimmed_indices = [i for i, _ in indexed_norms]
            indices_to_keep = untrimmed_indices[num_to_trim : -num_to_trim]
            print(f"PaillierTrimmedMean: Trimmed {num_to_trim} clients from each side, keeping {len(indices_to_keep)}.")
        # If no clients are left after trimming, stop.
        if not indices_to_keep:
            print("PaillierTrimmedMean: No clients left after trimming. Discarding update.")
            return None, {}
        
        # Get the encrypted updates from the selected clients using their original indices
        selected_client_updates_raw = [pickle.loads(parameters_to_ndarrays(results[i][1].parameters)[0].tobytes()) for i in indices_to_keep]

        # Homomorphically average the selected updates
        # Sum layers across selected clients
        summed_updates_raw = []
        if not selected_client_updates_raw: # Should be caught by previous check, but good for safety
            return None, {}
            
        first_client_update = selected_client_updates_raw[0]
        for i in range(len(first_client_update)): # Iterate through layers
            layer_sum = first_client_update[i]
            for j in range(1, len(selected_client_updates_raw)):
                layer_sum += selected_client_updates_raw[j][i]
            summed_updates_raw.append(layer_sum)

        aggregated_encrypted_vector = [layer_sum_array * (1.0 / len(selected_client_updates_raw)) for layer_sum_array in summed_updates_raw]
        
        # --- Measure server-side decryption time ---
        dec_start = time.perf_counter()
        decrypted_weights = decrypt_weights(self.p_context.private_key, aggregated_encrypted_vector)
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start
        # -------------------------------------------

        new_global_weights = decrypted_weights
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}


# ==============================================================================
# Paillier Metrics Strategy Wrapper
# ==============================================================================
class PaillierMetricsStrategyWrapper(Strategy):
    """A wrapper that sits on top of a Paillier strategy to collect and log metrics."""

    def __init__(self, base_strategy: Strategy, run_config: dict):
        super().__init__()
        self.base_strategy = base_strategy
        self.run_config = run_config
        self.run_name = run_config.get("run_name", "default_run")
        
        # State for tracking new metrics
        self.best_accuracy = 0.0
        self.convergence_round = -1 # -1 means not converged yet
        self.total_aggregation_time = 0.0
        self.total_client_encryption_time = 0.0  # Sum of client-reported encryption times
        self.total_server_decryption_time = 0.0  # Sum of server-side decryption times
        self.total_client_training_time = 0.0 # NEW: Initialize total client training time
        self.all_round_data = [] # To store detailed data for artifact

    def initialize_parameters(self, client_manager):
        return self.base_strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        # --- 1. Time the aggregation step ---
        start_time = time.perf_counter()
        aggregated_params, agg_metrics = self.base_strategy.aggregate_fit(server_round, results, failures)
        end_time = time.perf_counter()
        aggregation_time = end_time - start_time
        self.total_aggregation_time += aggregation_time

        # --- 2. Calculate communication costs and FHE times ---
        num_successful_clients = len(results)

        # Uplink: Sum of encrypted_size_bytes from all successful client results
        total_uplink_bytes = sum(res.metrics.get("encrypted_size_bytes", 0) for _, res in results)
        uplink_mb = total_uplink_bytes / (1024 * 1024)

        # Downlink: Size of the newly aggregated global model (sent plaintext from server)
        downlink_mb = 0
        if aggregated_params:
            downlink_bytes = get_weights_size_bytes(parameters_to_ndarrays(aggregated_params))
            downlink_mb = downlink_bytes / (1024 * 1024)

        # Paillier-specific metrics:
        # Sum of client-reported encryption times
        per_round_client_encryption_time = sum(res.metrics.get("encryption_time", 0) for _, res in results)
        self.total_client_encryption_time += per_round_client_encryption_time
        
        # Server-side decryption time (passed from base strategy's agg_metrics)
        per_round_server_decryption_time = agg_metrics.get("server_decryption_time", 0)
        self.total_server_decryption_time += per_round_server_decryption_time

        # Average expansion factor across participating clients
        expansion_factors = [res.metrics.get("expansion_factor", 0) for _, res in results]
        avg_expansion_factor = np.mean(expansion_factors) if expansion_factors else 0

        per_round_client_training_time = sum(res.metrics.get("training_time", 0) for _, res in results)
        self.total_client_training_time += per_round_client_training_time

        # --- 3. Log per-round metrics to W&B ---
        round_data = {
            "round": server_round,
            "per_round_aggregation_time": aggregation_time,
            "uplink_mb": uplink_mb,
            "downlink_mb": downlink_mb,
            "total_aggregation_time": self.total_aggregation_time,
            "per_round_client_encryption_time": per_round_client_encryption_time,
            "total_client_encryption_time": self.total_client_encryption_time,
            "per_round_server_decryption_time": per_round_server_decryption_time,
            "total_server_decryption_time": self.total_server_decryption_time,
            "per_round_avg_expansion_factor": avg_expansion_factor,
            "num_clients_participated": num_successful_clients,
            "num_clients_failed": len(failures),
            "per_round_client_training_time": per_round_client_training_time,
            "total_client_training_time": self.total_client_training_time,
        }
        wandb.log(round_data)
        
        # Store for final artifact
        self.all_round_data.append(round_data)

        return aggregated_params, agg_metrics

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        return self.base_strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round, parameters):
        # Use the base strategy's evaluate_fn
        loss, metrics = self.base_strategy.evaluate(server_round, parameters)
        accuracy = metrics["accuracy"]

        # --- Track Best Accuracy and Convergence Rate ---
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        
        threshold = self.run_config.get("convergence_threshold", 0.5)
        if accuracy >= threshold and self.convergence_round == -1:
            self.convergence_round = server_round

        # Log accuracy metrics to W&B
        wandb.log({
            "round": server_round,
            "server_loss": loss,
            "server_accuracy": accuracy,
            "best_accuracy_so_far": self.best_accuracy,
        })
        
        # Add accuracy to our round-by-round data store
        # Find the data for the current round and update it
        for rd in self.all_round_data:
            if rd["round"] == server_round:
                rd.update({"loss": loss, "accuracy": accuracy})
                break
        
        return loss, metrics

    def log_final_metrics_and_artifact(self):
        """Called at the very end of a simulation to log summary metrics and artifacts."""
        print("Logging final Paillier metrics and artifacts...")

        summary_metrics = {
            "final_best_accuracy": self.best_accuracy,
            "convergence_round": self.convergence_round,
            "final_total_aggregation_time": self.total_aggregation_time,
            "final_total_client_encryption_time": self.total_client_encryption_time,
            "final_total_server_decryption_time": self.total_server_decryption_time,
            "final_total_fhe_time": self.total_client_encryption_time + self.total_server_decryption_time,
            "final_total_client_training_time": self.total_client_training_time, # NEW: Add to summary
        }
        wandb.log(summary_metrics)

        run_dir = wandb.run.dir
        file_path = os.path.join(run_dir, f"{self.run_name}_paillier_rundata.json")

        with open(file_path, "w") as f:
            json.dump(self.all_round_data, f, indent=2)

        artifact = wandb.Artifact(name=f"{self.run_name}_paillier_details", type="dataset")
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)

        print(f"✅ Paillier Artifact '{self.run_name}_paillier_details' logged in {run_dir}")

# ==============================================================================
# The main builder function for ALL Paillier strategies with metrics wrapper
# ==============================================================================
def build_strategy_paillier(run_config: dict) -> Tuple[Strategy, PaillierContext]: # Changed return type
    strategy_name = run_config.get("strategy")
    num_malicious = run_config.get("num_malicious", 0)
    run_name = run_config.get("run_name", "default_paillier_run")

    wandb.init(project="fl-paillier-benchmark", name=run_name, reinit=True, settings=wandb.Settings(start_method="thread"))
    wandb.config.update(run_config)

    p_context = PaillierContext(key_size=512) # Use default key_size from task_paillier
    
    # Using SmallNet for all Paillier runs to ensure reasonable performance
    model_to_use = SmallNet()
    print(f"✅ Server using model: {model_to_use.__class__.__name__}")

    initial_parameters = ndarrays_to_parameters(get_weights(model_to_use))
    
    testloader = get_central_testloader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round: int, parameters: List[np.ndarray], _):
        set_weights(model_to_use, parameters)
        loss, acc = test(model_to_use, testloader, device)
        return loss, {"accuracy": acc}

    def fit_config(server_round: int):
        config_to_send = run_config.copy()
        config_to_send["round"] = server_round
        return config_to_send # No public key needed here, clients get it from p_context directly
    
    common_args = {
        "paillier_context": p_context,
        "sample_model": model_to_use,
        "fraction_fit": 1.0, "fraction_evaluate": 0.0,
        "min_available_clients": run_config["num_partitions"],
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate, "on_fit_config_fn": fit_config,
    }
    
    base_strategy: Strategy
    if strategy_name == "PaillierTrimmedMean":
        base_strategy = PaillierTrimmedMean(num_malicious_clients=num_malicious, **common_args)
    else: # Default to PaillierFedAvg
        base_strategy = PaillierFedAvg(**common_args)

    # Wrap the base strategy with metrics collection
    wrapped_strategy = PaillierMetricsStrategyWrapper(base_strategy=base_strategy, run_config=run_config)

    return wrapped_strategy, p_context # Return the PaillierContext object
