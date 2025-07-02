# --- START OF FILE server_app.py ---

import wandb
import torch
import tempfile
import os
import json # [NEW] Use JSON for artifacts, it's more portable than pickle
import time
import numpy as np
from typing import List, Tuple, Union, Optional, Dict

from flwr.server.strategy.aggregate import aggregate
from flwr.server.strategy import FedAvg, Krum, Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
# [MODIFIED] Import the new size calculation helper
from .task import SmallNet, get_weights, get_central_testloader, set_weights, test, flatten_weights, get_weights_size_bytes

# ... (MultiKrum, TrimmedMean, Bulyan class implementations remain unchanged) ...
# In sec_agg/server_app.py

class MultiKrum(Krum):
    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}
        weights_results = [(parameters_to_ndarrays(res.parameters), res.num_examples) for _, res in results]
        
        # ... (rest of the distance and score calculation logic is the same) ...
        flat_weights = [flatten_weights(w) for w, _ in weights_results]
        distances = []
        for i, p_i in enumerate(flat_weights):
            dists_i = []
            for j, p_j in enumerate(flat_weights):
                if i == j: continue
                dists_i.append(np.linalg.norm(p_i - p_j) ** 2)
            distances.append(dists_i)

        m = len(weights_results) - self.num_malicious_clients - 2
        scores = [sum(sorted(dists)[:m]) for dists in distances]
        best_indices = np.argsort(scores)[:self.num_clients_to_keep]
        best_results = [weights_results[i] for i in best_indices]

        # [NEW] Add this safety check
        if not best_results:
            print("MultiKrum: No clients left after selection. Discarding update.")
            return None, {}

        return ndarrays_to_parameters(aggregate(best_results)), {}

# In sec_agg/server_app.py

# In sec_agg/server_app.py

class TrimmedMean(FedAvg):
    """
    An implementation of the Trimmed Mean robust aggregation rule.
    It discards a certain number of clients with the highest and lowest
    update norms before averaging the rest.
    """
    def __init__(self, num_malicious_clients: int, **kwargs):
        super().__init__(**kwargs)
        self.num_to_trim_each_end = num_malicious_clients

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        # 1. Get client weights and calculate their norms
        samples = [(parameters_to_ndarrays(res.parameters), res.num_examples) for _, res in results]
        norms = [np.linalg.norm(flatten_weights(params)) for params, _ in samples]
        
        # 2. Sort clients based on their norms
        indexed_norms = sorted(enumerate(norms), key=lambda x: x[1])
        sorted_indices = [i for i, _ in indexed_norms]

        # 3. Determine which clients to keep
        num_clients_received = len(sorted_indices)
        num_to_trim = self.num_to_trim_each_end

        if num_clients_received <= 2 * num_to_trim:
            # Not enough clients to trim, so we keep all of them
            print(f"TrimmedMean: Not enough clients ({num_clients_received}) to trim {num_to_trim} from each end. Keeping all.")
            indices_to_keep = sorted_indices
        else:
            # Perform the trimming
            indices_to_keep = sorted_indices[num_to_trim : num_clients_received - num_to_trim]

        # 4. Aggregate the selected clients
        samples_to_aggregate = [samples[i] for i in indices_to_keep]

        # Safety check: if after everything, we have nothing, discard the update
        if not samples_to_aggregate:
            print("TrimmedMean: No clients left after trimming. Discarding update.")
            return None, {}

        print(f"TrimmedMean: Trimmed {num_clients_received - len(samples_to_aggregate)} clients, aggregating {len(samples_to_aggregate)}.")
        
        aggregated_ndarrays = aggregate(samples_to_aggregate)
        return ndarrays_to_parameters(aggregated_ndarrays), {}

class Bulyan(FedAvg):
    def __init__(self, num_malicious_clients: int, selection_size: int, trimmed_mean_beta: int, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious_clients = num_malicious_clients
        self.selection_size = selection_size
        self.trimmed_mean_beta = trimmed_mean_beta

    def _krum_selection(self, client_updates: List[np.ndarray]) -> List[np.ndarray]:
        flat_updates = [flatten_weights(u) for u in client_updates]
        num_clients = len(flat_updates)
        
        distances = []
        for i in range(num_clients):
            dists_i = []
            for j in range(num_clients):
                if i == j: continue
                dists_i.append(np.linalg.norm(flat_updates[i] - flat_updates[j]) ** 2)
            distances.append(dists_i)

        m = num_clients - self.num_malicious_clients - 2
        scores = [sum(sorted(dists)[:m]) for dists in distances]
        
        best_indices = np.argsort(scores)[:self.selection_size]
        return [client_updates[i] for i in best_indices]

    def _trimmed_mean_aggregation(self, selected_updates: List[np.ndarray]) -> List[np.ndarray]:
        if not selected_updates: return []
        
        num_layers = len(selected_updates[0])
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            layer_updates = np.stack([client[layer_idx] for client in selected_updates])
            
            num_clients_in_layer = layer_updates.shape[0]
            if num_clients_in_layer <= 2 * self.trimmed_mean_beta:
                trimmed_mean_layer = np.mean(layer_updates, axis=0)
            else:
                sorted_updates = np.sort(layer_updates, axis=0)
                trimmed_updates = sorted_updates[self.trimmed_mean_beta : -self.trimmed_mean_beta]
                trimmed_mean_layer = np.mean(trimmed_updates, axis=0)
            
            aggregated_weights.append(trimmed_mean_layer)
        
        return aggregated_weights

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}
        
        client_updates = [parameters_to_ndarrays(res.parameters) for _, res in results]
        selected_updates = self._krum_selection(client_updates)
        if not selected_updates: return None, {}
        aggregated_weights = self._trimmed_mean_aggregation(selected_updates)
        if not aggregated_weights: return None, {}
        
        return ndarrays_to_parameters(aggregated_weights), {}

# ==============================================================================
# [NEW] 1. Define the Metrics Strategy Wrapper
# ==============================================================================
class MetricsStrategyWrapper(Strategy):
    """A wrapper that sits on top of a base strategy to collect and log metrics."""

    def __init__(self, base_strategy: Strategy, run_config: dict):
        super().__init__()
        self.base_strategy = base_strategy
        self.run_config = run_config
        self.run_name = run_config.get("run_name", "default_run")
        
        # State for tracking new metrics
        self.best_accuracy = 0.0
        self.convergence_round = -1 # -1 means not converged yet
        self.total_aggregation_time = 0.0
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

        # --- 2. Calculate communication costs ---
        # Uplink: Sum of bytes from all successful client results
        total_uplink_bytes = sum(res.metrics.get("uplink_bytes", 0) for _, res in results)
        uplink_mb = total_uplink_bytes / (1024 * 1024)

        # Downlink: Size of the newly aggregated global model
        downlink_mb = 0
        if aggregated_params:
            downlink_bytes = get_weights_size_bytes(parameters_to_ndarrays(aggregated_params))
            downlink_mb = downlink_bytes / (1024 * 1024)

        # --- 3. Log per-round metrics to W&B ---
        round_data = {
            "round": server_round,
            "per_round_aggregation_time": aggregation_time,
            "uplink_mb": uplink_mb,
            "downlink_mb": downlink_mb,
            "total_aggregation_time": self.total_aggregation_time,
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
        print("Logging final metrics and artifacts...")
        summary_metrics = {
            "final_best_accuracy": self.best_accuracy,
            "convergence_round": self.convergence_round,
            "final_total_aggregation_time": self.total_aggregation_time,
        }
        wandb.log(summary_metrics)

        # Create and log an artifact with all the detailed round-by-round data
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, f"{self.run_name}_rundata.json")
            with open(file_path, "w") as f:
                json.dump(self.all_round_data, f, indent=2)
            
            artifact = wandb.Artifact(name=f"{self.run_name}_details", type="dataset")
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
            print(f"✅ Artifact '{self.run_name}_details' logged.")


# ==============================================================================
# [MODIFIED] 2. The main builder function now uses the wrapper
# ==============================================================================
def build_strategy(run_config: dict) -> Strategy:
    run_name = run_config.get("run_name", "default_run")
    strategy_name = run_config.get("strategy", "FedAvg")
    num_malicious = run_config.get("num_malicious", 0)
    local_epochs = run_config.get("local_epochs", 1)

    wandb.init(project="fl-robustness-evaluation", name=run_name, reinit=True,
               settings=wandb.Settings(start_method="thread"), config=run_config)

    net = SmallNet()
    initial_parameters = ndarrays_to_parameters(get_weights(net))
    
    testloader = get_central_testloader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # [MODIFIED] This is now a pure function, logging is handled by the wrapper.
    def evaluate_fn(_: int, parameters: List[np.ndarray], __) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # ### THIS IS THE CRITICAL FIX ###
        # The 'parameters' variable is ALREADY a list of numpy arrays here.
        # We must not call parameters_to_ndarrays() on it again.
        set_weights(net, parameters) 
        
        loss, acc = test(net, testloader, device)
        return loss, {"accuracy": acc}

    def fit_config(server_round):
        return {"local_epochs": local_epochs, "round": server_round}
    
    # --- Instantiate the base strategy ---
    base_strategy: Strategy
    common_args = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 0.0, # Evaluate on all clients (not used by default eval_fn)
        "min_fit_clients": run_config["num_partitions"] - num_malicious,
        "min_available_clients": run_config["num_partitions"],
        "evaluate_fn": evaluate_fn, # The corrected function is passed here
        "on_fit_config_fn": fit_config,
        "initial_parameters": initial_parameters,
    }
    
    if strategy_name == "FedAvg":
        base_strategy = FedAvg(**common_args)
    elif strategy_name == "Krum":
        base_strategy = Krum(num_malicious_clients=num_malicious, **common_args)
    elif strategy_name == "MultiKrum":
        base_strategy = MultiKrum(
            num_malicious_clients=num_malicious, 
            num_clients_to_keep=run_config["num_partitions"] - num_malicious,
            **common_args
        )
    elif strategy_name == "TrimmedMean":
        base_strategy = TrimmedMean(num_malicious_clients=num_malicious, **common_args)
    elif strategy_name == "Bulyan":
        selection_size = run_config.get("bulyan_selection_size", run_config["num_partitions"] - num_malicious)
        trimmed_mean_beta = run_config.get("trimmed_mean_beta", 0)
        base_strategy = Bulyan(
            num_malicious_clients=num_malicious,
            selection_size=selection_size,
            trimmed_mean_beta=trimmed_mean_beta,
            **common_args
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
        
    # --- Wrap the base strategy with our metrics collector ---
    return MetricsStrategyWrapper(base_strategy=base_strategy, run_config=run_config)