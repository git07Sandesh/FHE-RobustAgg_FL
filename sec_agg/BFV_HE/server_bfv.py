# sec_agg/server_bfv.py

import wandb
import torch
import time
import numpy as np
import os
import json
from typing import List, Tuple, Optional, Dict, Union # Added Union for failures

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
from sec_agg.Plaintext.task import Net, SmallNet, get_central_testloader, get_weights, set_weights, test, unflatten_weights, flatten_weights, get_weights_size_bytes
from .task_bfv import get_bfv_context, decode

# ==============================================================================
# BFV Strategy Implementations
# ==============================================================================

class BFVFedAvg(FedAvg):
    def __init__(self, fhe_context: ts.Context, sample_model: torch.nn.Module, precision_bits: int, **kwargs):
        super().__init__(**kwargs)
        self.fhe_context = fhe_context
        self.sample_model = sample_model
        self.precision_bits = precision_bits

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]], # Changed any to Union
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}
        
        num_examples_list = [res.num_examples for _, res in results]
        total_examples = sum(num_examples_list)
        if total_examples == 0: return None, {}

        # Decrypt all client updates
        # This decryption is implicitly part of the overall aggregation_time measured by the wrapper.
        # We will explicitly measure the *final* model decryption.
        client_updates_encoded = []
        for _, res in results:
            uint8_array = parameters_to_ndarrays(res.parameters)[0]
            bfv_vector = ts.bfv_vector_from(self.fhe_context, uint8_array.tobytes())
            client_updates_encoded.append(np.array(bfv_vector.decrypt()))

        # Perform weighted average on the plaintext encoded integers
        weighted_encoded_updates = [update * n_ex for update, n_ex in zip(client_updates_encoded, num_examples_list)]
        summed_weighted_updates = np.sum(np.array(weighted_encoded_updates), axis=0)
        averaged_encoded_vector = np.round(summed_weighted_updates / total_examples).astype(np.int64)

        # --- Measure server-side decryption time (final result) ---
        # Note: BFVFedAvg directly uses the decrypted average, so the decryption time is effectively
        # distributed among the initial decryptions. For consistency with CKKS and to measure
        # the 'final model' decryption, we'll decode here and consider it the "final decryption" step.
        dec_start = time.perf_counter()
        decoded_flat_weights = decode(averaged_encoded_vector.tolist(), self.precision_bits)
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start
        # -----------------------------------------------------------
        
        new_global_weights = unflatten_weights(decoded_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}


class BFVTrimmedMean(FedAvg):
    def __init__(self, num_malicious_clients: int, fhe_context: ts.Context, sample_model: torch.nn.Module, precision_bits: int, **kwargs):
        super().__init__(**kwargs)
        self.b = num_malicious_clients
        self.fhe_context = fhe_context
        self.sample_model = sample_model
        self.precision_bits = precision_bits

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}

        # "Leaky" protocol: Get plaintext L2 norms from client metrics
        norms = [res.metrics.get("l2_norm", 0.0) for _, res in results]
        
        indexed_norms = sorted(enumerate(norms), key=lambda x: x[1])
        
        num_to_trim = self.b
        num_clients_available = len(indexed_norms)
        
        if num_to_trim == 0:
                # If no malicious clients, no trimming should occur; keep all.
            indices_to_keep = list(range(num_clients_available))
            print(f"TrimmedMean: Benign scenario, keeping all {len(indices_to_keep)} clients.")
        elif num_clients_available <= 2 * num_to_trim:
            # If not enough clients to trim from both sides, or trimming would remove too many,
            # (e.g., if N=10, b=4, then 2*b=8. N=10 > 2*b=8. If N=7, b=4, then 2*b=8. N=7 <= 2*b=8)
            # In such cases, standard Trimmed Mean might fail or require a fallback.
            # For now, we'll keep all as a safer default than discarding completely.
            indices_to_keep = list(range(num_clients_available))
            print(f"Warning: TrimmedMean intended to trim {num_to_trim} from each side but only {num_clients_available} clients available. Keeping all clients due to insufficient count.")
        else:
            # Standard trimming: remove 'num_to_trim' from each end
            untrimmed_indices = [i for i, _ in indexed_norms]
            indices_to_keep = untrimmed_indices[num_to_trim : -num_to_trim]
            print(f"TrimmedMean: Trimmed {num_to_trim} clients from each side, keeping {len(indices_to_keep)}.")
                   
        # Safety check: if no clients are kept, return None
        if not indices_to_keep:
            print("BFVTrimmedMean: No clients left after trimming. Discarding update.")
            return None, {}

        # Only decrypt the selected updates
        selected_updates_encoded = []
        for i in indices_to_keep:
            uint8_array = parameters_to_ndarrays(results[i][1].parameters)[0]
            bfv_vector = ts.bfv_vector_from(self.fhe_context, uint8_array.tobytes())
            selected_updates_encoded.append(np.array(bfv_vector.decrypt()))
        
        if not selected_updates_encoded: return None, {}
        
        # Use np.mean directly on the array for robustness, then round and cast
        averaged_encoded_vector = np.mean(np.array(selected_updates_encoded), axis=0).astype(np.int64)

        # --- Measure server-side decryption time (final result) ---
        dec_start = time.perf_counter()
        decoded_flat_weights = decode(averaged_encoded_vector.tolist(), self.precision_bits)
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start
        # -----------------------------------------------------------

        new_global_weights = unflatten_weights(decoded_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}

# ==============================================================================
# BFV Metrics Strategy Wrapper
# ==============================================================================
class BFVMetricsStrategyWrapper(Strategy):
    """A wrapper that sits on top of a BFV strategy to collect and log metrics."""

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
        self.total_client_training_time = 0.0
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

        # Downlink: Size of the newly aggregated global model
        downlink_mb = 0
        if aggregated_params:
            downlink_bytes = get_weights_size_bytes(parameters_to_ndarrays(aggregated_params))
            downlink_mb = downlink_bytes / (1024 * 1024)

        # BFV-specific metrics:
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
        print("Logging final BFV metrics and artifacts...")

        summary_metrics = {
            "final_best_accuracy": self.best_accuracy,
            "convergence_round": self.convergence_round,
            "final_total_aggregation_time": self.total_aggregation_time,
            "final_total_client_encryption_time": self.total_client_encryption_time,
            "final_total_server_decryption_time": self.total_server_decryption_time,
            "final_total_fhe_time": self.total_client_encryption_time + self.total_server_decryption_time,
            "final_total_client_training_time": self.total_client_training_time,
        }
        wandb.log(summary_metrics)

        run_dir = wandb.run.dir
        file_path = os.path.join(run_dir, f"{self.run_name}_bfv_rundata.json")

        with open(file_path, "w") as f:
            json.dump(self.all_round_data, f, indent=2)

        artifact = wandb.Artifact(name=f"{self.run_name}_bfv_details", type="dataset")
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)

        print(f"✅ BFV Artifact '{self.run_name}_bfv_details' logged in {run_dir}")

# ==============================================================================
# The main builder function for ALL BFV strategies with metrics wrapper
# ==============================================================================
def build_strategy_bfv(run_config: dict) -> Tuple[Strategy, bytes]: # Changed return type for context
    strategy_name = run_config.get("strategy")
    num_malicious = run_config.get("num_malicious", 0)
    run_name = run_config.get("run_name", "default_bfv_run")
    
    wandb.init(project="fl-fhe-bfv-benchmark", name=run_name, reinit=True, settings=wandb.Settings(start_method="thread"))
    wandb.config.update(run_config)

    fhe_context = get_bfv_context()
    context_bytes = fhe_context.serialize(save_secret_key=False) # Serialize context for clients

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
        config_to_send["fhe_context_bytes"] = context_bytes # Pass context bytes to clients
        return config_to_send

    common_args = {
        "fhe_context": fhe_context,
        "sample_model": model_to_use,
        "fraction_fit": 1.0,
        "fraction_evaluate": 0.0,
        "min_available_clients": run_config["num_partitions"],
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate,
        "on_fit_config_fn": fit_config,
        "precision_bits": 16 # Default precision bits
    }

    # Create the base strategy
    base_strategy: Strategy
    if strategy_name == "BFVTrimmedMean":
        base_strategy = BFVTrimmedMean(num_malicious_clients=num_malicious, **common_args)
    else: # Default to BFVFedAvg
        base_strategy = BFVFedAvg(**common_args)

    # Wrap the base strategy with metrics collection
    wrapped_strategy = BFVMetricsStrategyWrapper(base_strategy=base_strategy, run_config=run_config)

    return wrapped_strategy, context_bytes # Return serialized context
