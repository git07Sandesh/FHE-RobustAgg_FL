# sec_agg/server_ckks.py

import wandb
import torch
import time
import tenseal as ts
import numpy as np
import os
import json
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
from sec_agg.Plaintext.task import Net, SmallNet, get_central_testloader, get_weights, set_weights, test, unflatten_weights, flatten_weights, get_weights_size_bytes
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
        dec_start = time.perf_counter()
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start
        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}
    
class CKKSKrum(FedAvg):
    def __init__(self, num_malicious_clients: int, fhe_context: ts.Context, sample_model: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious_clients = num_malicious_clients
        self.fhe_context = fhe_context
        self.sample_model = sample_model

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}

        client_vectors = [ts.ckks_vector_from(self.fhe_context, parameters_to_ndarrays(res.parameters)[0].tobytes()) for _, res in results]
        n_clients = len(client_vectors)

        print("Homomorphically computing pairwise squared distances for Krum...")
        scores = []
        for i in range(n_clients):
            distances = []
            # For Krum, we typically sum distances to all other clients
            for j in range(n_clients):
                if i == j: continue
                distance_vec = client_vectors[i] - client_vectors[j]
                squared_distance = distance_vec.dot(distance_vec) 
                distances.append(squared_distance)
            scores.append(sum(distances))
        
        print("Decrypting Krum scores...")
        decrypted_scores = [score.decrypt()[0] for score in scores]
        
        # Find the client with the minimum score (most similar to others)
        best_client_idx = np.argmin(decrypted_scores)
        print(f"Krum selected client with index: {best_client_idx}")
        
        # Use only the selected client's update
        selected_vector = client_vectors[best_client_idx]
        
        dec_start = time.perf_counter()
        decrypted_flat_weights = np.array(selected_vector.decrypt())
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start

        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}
    
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
        
        # Safety check: if no clients are selected, return None
        if not indices_to_keep:
            print("CKKSMultiKrum: No clients left after selection. Discarding update.")
            return None, {}
        
        selected_updates = [client_vectors[i] for i in indices_to_keep]
        aggregated_vector = sum(selected_updates) * (1 / len(selected_updates))
        
        dec_start = time.perf_counter()
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start


        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}

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
            print("CKKSTrimmedMean: No clients left after trimming. Discarding update.")
            return None, {}
        
        # Get encrypted vectors from the original results list using the kept indices
        client_vectors = [ts.ckks_vector_from(self.fhe_context, parameters_to_ndarrays(results[i][1].parameters)[0].tobytes()) for i in indices_to_keep]
        
        if not client_vectors: return None, {}

        aggregated_vector = sum(client_vectors) * (1 / len(client_vectors))
        
        dec_start = time.perf_counter()
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start

        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}
  
class CKKSBulyan(FedAvg):
    def __init__(self, num_malicious_clients: int, fhe_context: ts.Context, sample_model: torch.nn.Module, 
                 bulyan_selection_size: Optional[int] = None, trimmed_mean_beta: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious_clients = num_malicious_clients
        self.fhe_context = fhe_context
        self.sample_model = sample_model
        self.bulyan_selection_size = bulyan_selection_size
        self.trimmed_mean_beta = trimmed_mean_beta if trimmed_mean_beta is not None else num_malicious_clients

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}

        client_vectors = [ts.ckks_vector_from(self.fhe_context, parameters_to_ndarrays(res.parameters)[0].tobytes()) for _, res in results]
        n_clients = len(client_vectors)
        f = self.num_malicious_clients
        
        # Bulyan parameters
        theta = self.bulyan_selection_size if self.bulyan_selection_size is not None else n_clients - 2 * f
        beta = self.trimmed_mean_beta
        
        print(f"Bulyan parameters: theta={theta}, beta={beta}, n_clients={n_clients}, f={f}")
        
        if theta <= 0:
            print(f"Warning: Bulyan requires n > 2f, but got n={n_clients}, f={f}")
            # Fall back to simple averaging
            aggregated_vector = sum(client_vectors) * (1 / len(client_vectors))
        else:
            print(f"Bulyan Phase 1: Selecting {theta} clients out of {n_clients} using iterative Krum selection...")
            
            # Phase 1: Iterative Krum selection
            remaining_indices = list(range(n_clients))
            remaining_vectors = client_vectors.copy()
            selected_indices = []
            
            for selection_round in range(theta):
                if len(remaining_vectors) <= 1:
                    break
                    
                print(f"Krum selection round {selection_round + 1}/{theta}")
                
                # Compute Krum scores for remaining clients
                scores = []
                for i in range(len(remaining_vectors)):
                    distances = []
                    for j in range(len(remaining_vectors)):
                        if i == j: continue
                        distance_vec = remaining_vectors[i] - remaining_vectors[j]
                        squared_distance = distance_vec.dot(distance_vec)
                        distances.append(squared_distance)
                    scores.append(sum(distances))
                
                # Decrypt scores and find the best client
                decrypted_scores = [score.decrypt()[0] for score in scores]
                best_local_idx = np.argmin(decrypted_scores)
                best_global_idx = remaining_indices[best_local_idx]
                
                # Select this client and remove from remaining
                selected_indices.append(best_global_idx)
                remaining_indices.pop(best_local_idx)
                remaining_vectors.pop(best_local_idx)
            
            print(f"Bulyan Phase 1 selected clients: {selected_indices}")
            
            # Phase 2: Coordinate-wise trimmed mean using L2 norms (same as TrimmedMean)
            print(f"Bulyan Phase 2: Computing trimmed mean on {len(selected_indices)} vectors with beta={beta}...")
            
            # Get L2 norms from client metrics for the selected clients
            selected_norms = [results[i][1].metrics.get("l2_norm", 0.0) for i in selected_indices]
            
            # Sort by norms and trim beta smallest and beta largest
            indexed_norms = sorted(enumerate(selected_norms), key=lambda x: x[1])
            
            if len(indexed_norms) <= 2 * beta:
                # Not enough vectors to trim, use all selected vectors
                indices_to_keep = list(range(len(selected_indices)))
                print(f"Bulyan Phase 2: Not enough vectors to trim (need >{2*beta}, have {len(selected_indices)}), using all selected vectors")
            else:
                # Trim beta from each end
                indices_to_keep = [i for i, _ in indexed_norms[beta:-beta]]
                print(f"Bulyan Phase 2: Trimmed {beta} vectors from each end, keeping {len(indices_to_keep)} vectors")
            
            # Map back to the original selected indices
            final_selected_indices = [selected_indices[i] for i in indices_to_keep]
            final_vectors = [client_vectors[i] for i in final_selected_indices]
            
            print(f"Bulyan final selected clients: {final_selected_indices}")
            
            # Safety check: if no vectors remain, fall back to simple average
            if not final_vectors:
                print("Warning: No vectors remaining after Bulyan, falling back to simple average")
                aggregated_vector = sum(client_vectors) * (1 / len(client_vectors))
            else:
                # Final aggregation
                aggregated_vector = sum(final_vectors) * (1 / len(final_vectors))
        
        dec_start = time.perf_counter()
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start

        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}


# ==============================================================================
# CKKS Metrics Strategy Wrapper
# ==============================================================================
class CKKSMetricsStrategyWrapper(Strategy):
    """A wrapper that sits on top of a CKKS strategy to collect and log metrics."""

    def __init__(self, base_strategy: Strategy, run_config: dict):
        super().__init__()
        self.base_strategy = base_strategy
        self.run_config = run_config
        self.run_name = run_config.get("run_name", "default_run")
        
        # State for tracking new metrics
        self.best_accuracy = 0.0
        self.convergence_round = -1 # -1 means not converged yet
        self.total_aggregation_time = 0.0
        self.total_client_encryption_time = 0.0  # CKKS-specific metric
        self.total_server_decryption_time = 0.0  # CKKS-specific metric
        self.total_client_training_time = 0.0 # NEW: Initialize total client training time
        self.all_round_data = [] # To store detailed data for artifact

    def initialize_parameters(self, client_manager):
        return self.base_strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],): # Type hints for clarity
        # --- 1. Time the aggregation step ---
        start_time = time.perf_counter()
        aggregated_params, agg_metrics = self.base_strategy.aggregate_fit(server_round, results, failures)
        end_time = time.perf_counter()
        aggregation_time = end_time - start_time
        self.total_aggregation_time += aggregation_time

        num_successful_clients = len(results)

        # --- 2. Calculate communication costs ---
        # Uplink: Sum of bytes from all successful client results
        total_uplink_kb = sum(res.metrics.get("uplink_kb", 0) for _, res in results)
        uplink_mb = total_uplink_kb / 1024 # Convert KB to MB

        # Downlink: Size of the newly aggregated global model
        downlink_mb = 0
        if aggregated_params:
            downlink_bytes = get_weights_size_bytes(parameters_to_ndarrays(aggregated_params))
            downlink_mb = downlink_bytes / (1024 * 1024)

        # --- 3. CKKS-specific metrics ---
        # Sum of client-reported encryption times
        per_round_client_encryption_time = sum(res.metrics.get("encryption_time", 0) for _, res in results)
        self.total_client_encryption_time += per_round_client_encryption_time
        
        # Server-side decryption time (passed from base strategy's agg_metrics)
        per_round_server_decryption_time = agg_metrics.get("server_decryption_time", 0)
        self.total_server_decryption_time += per_round_server_decryption_time

        # Average expansion factor across participating clients
        expansion_factors = [res.metrics.get("expansion_factor", 0) for _, res in results]
        avg_expansion_factor = np.mean(expansion_factors) if expansion_factors else 0

        # NEW: Extract and sum client training time
        # This will give the sum of training times across all participating clients in this round.
        per_round_client_training_time = sum(res.metrics.get("training_time", 0) for _, res in results)
        self.total_client_training_time += per_round_client_training_time


        # --- 4. Log per-round metrics to W&B ---
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
            # NEW: Add training time metrics
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
        print("Logging final CKKS metrics and artifacts...")

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
        file_path = os.path.join(run_dir, f"{self.run_name}_ckks_rundata.json")

        with open(file_path, "w") as f:
            json.dump(self.all_round_data, f, indent=2)

        artifact = wandb.Artifact(name=f"{self.run_name}_ckks_details", type="dataset")
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)

        print(f"✅ CKKS Artifact '{self.run_name}_ckks_details' logged in {run_dir}")


# ==============================================================================
# The main builder function for ALL CKKS strategies with metrics wrapper
# ==============================================================================
def build_strategy_ckks(run_config: dict) -> Tuple[Strategy, bytes]:
    strategy_name = run_config.get("strategy")
    num_malicious = run_config.get("num_malicious", 0)
    run_name = run_config.get("run_name", "default_ckks_run")

    wandb.init(project="fl-fhe-ckks-benchmark", name=run_name, reinit=True, settings=wandb.Settings(start_method="thread"))
    wandb.config.update(run_config)

    fhe_context = get_ckks_context()
    context_bytes = fhe_context.serialize(save_secret_key=False)

    # MultiKrum requires the smaller model for FHE dot products.
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
        config_to_send["fhe_context_bytes"] = context_bytes
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
    }
    
    # Create the base strategy
    base_strategy: Strategy
    if strategy_name == "CKKSKrum":
        base_strategy = CKKSKrum(num_malicious_clients=num_malicious, **common_args)
    elif strategy_name == "CKKSMultiKrum":
        base_strategy = CKKSMultiKrum(num_malicious_clients=num_malicious, num_clients_to_keep=run_config["num_partitions"] - num_malicious, **common_args)
    elif strategy_name == "CKKSTrimmedMean":
        base_strategy = CKKSTrimmedMean(num_malicious_clients=num_malicious, **common_args)
    elif strategy_name == "CKKSBulyan":
        selection_size = run_config.get("bulyan_selection_size")
        trimmed_mean_beta = run_config.get("trimmed_mean_beta")
        
        base_strategy = CKKSBulyan(
            num_malicious_clients=num_malicious,
            bulyan_selection_size=selection_size,
            trimmed_mean_beta=trimmed_mean_beta,
            **common_args
        )
    else: # Default to CKKSFedAvg
        base_strategy = CKKSFedAvg(**common_args)

    # Wrap the base strategy with metrics collection
    wrapped_strategy = CKKSMetricsStrategyWrapper(base_strategy=base_strategy, run_config=run_config)

    return wrapped_strategy, context_bytes# sec_agg/server_ckks.py

import wandb
import torch
import time
import tenseal as ts
import numpy as np
import os
import json
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
from sec_agg.Plaintext.task import Net, SmallNet, get_central_testloader, get_weights, set_weights, test, unflatten_weights, flatten_weights, get_weights_size_bytes
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
        dec_start = time.perf_counter()
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start
        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}
    
class CKKSKrum(FedAvg):
    def __init__(self, num_malicious_clients: int, fhe_context: ts.Context, sample_model: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious_clients = num_malicious_clients
        self.fhe_context = fhe_context
        self.sample_model = sample_model

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}

        client_vectors = [ts.ckks_vector_from(self.fhe_context, parameters_to_ndarrays(res.parameters)[0].tobytes()) for _, res in results]
        n_clients = len(client_vectors)

        print("Homomorphically computing pairwise squared distances for Krum...")
        scores = []
        for i in range(n_clients):
            distances = []
            # For Krum, we typically sum distances to all other clients
            for j in range(n_clients):
                if i == j: continue
                distance_vec = client_vectors[i] - client_vectors[j]
                squared_distance = distance_vec.dot(distance_vec) 
                distances.append(squared_distance)
            scores.append(sum(distances))
        
        print("Decrypting Krum scores...")
        decrypted_scores = [score.decrypt()[0] for score in scores]
        
        # Find the client with the minimum score (most similar to others)
        best_client_idx = np.argmin(decrypted_scores)
        print(f"Krum selected client with index: {best_client_idx}")
        
        # Use only the selected client's update
        selected_vector = client_vectors[best_client_idx]
        
        dec_start = time.perf_counter()
        decrypted_flat_weights = np.array(selected_vector.decrypt())
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start

        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}
    
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
        
        # Safety check: if no clients are selected, return None
        if not indices_to_keep:
            print("CKKSMultiKrum: No clients left after selection. Discarding update.")
            return None, {}
        
        selected_updates = [client_vectors[i] for i in indices_to_keep]
        aggregated_vector = sum(selected_updates) * (1 / len(selected_updates))
        
        dec_start = time.perf_counter()
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start


        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}

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
            print("CKKSTrimmedMean: No clients left after trimming. Discarding update.")
            return None, {}
        
        # Get encrypted vectors from the original results list using the kept indices
        client_vectors = [ts.ckks_vector_from(self.fhe_context, parameters_to_ndarrays(results[i][1].parameters)[0].tobytes()) for i in indices_to_keep]
        
        if not client_vectors: return None, {}

        aggregated_vector = sum(client_vectors) * (1 / len(client_vectors))
        
        dec_start = time.perf_counter()
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start

        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}
  
class CKKSBulyan(FedAvg):
    def __init__(self, num_malicious_clients: int, fhe_context: ts.Context, sample_model: torch.nn.Module, 
                 bulyan_selection_size: Optional[int] = None, trimmed_mean_beta: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious_clients = num_malicious_clients
        self.fhe_context = fhe_context
        self.sample_model = sample_model
        self.bulyan_selection_size = bulyan_selection_size
        self.trimmed_mean_beta = trimmed_mean_beta if trimmed_mean_beta is not None else num_malicious_clients

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}

        client_vectors = [ts.ckks_vector_from(self.fhe_context, parameters_to_ndarrays(res.parameters)[0].tobytes()) for _, res in results]
        n_clients = len(client_vectors)
        f = self.num_malicious_clients
        
        # Bulyan parameters
        theta = self.bulyan_selection_size if self.bulyan_selection_size is not None else n_clients - 2 * f
        beta = self.trimmed_mean_beta
        
        print(f"Bulyan parameters: theta={theta}, beta={beta}, n_clients={n_clients}, f={f}")
        
        if theta <= 0:
            print(f"Warning: Bulyan requires n > 2f, but got n={n_clients}, f={f}")
            # Fall back to simple averaging
            aggregated_vector = sum(client_vectors) * (1 / len(client_vectors))
        else:
            print(f"Bulyan Phase 1: Selecting {theta} clients out of {n_clients} using iterative Krum selection...")
            
            # Phase 1: Iterative Krum selection
            remaining_indices = list(range(n_clients))
            remaining_vectors = client_vectors.copy()
            selected_indices = []
            
            for selection_round in range(theta):
                if len(remaining_vectors) <= 1:
                    break
                    
                print(f"Krum selection round {selection_round + 1}/{theta}")
                
                # Compute Krum scores for remaining clients
                scores = []
                for i in range(len(remaining_vectors)):
                    distances = []
                    for j in range(len(remaining_vectors)):
                        if i == j: continue
                        distance_vec = remaining_vectors[i] - remaining_vectors[j]
                        squared_distance = distance_vec.dot(distance_vec)
                        distances.append(squared_distance)
                    scores.append(sum(distances))
                
                # Decrypt scores and find the best client
                decrypted_scores = [score.decrypt()[0] for score in scores]
                best_local_idx = np.argmin(decrypted_scores)
                best_global_idx = remaining_indices[best_local_idx]
                
                # Select this client and remove from remaining
                selected_indices.append(best_global_idx)
                remaining_indices.pop(best_local_idx)
                remaining_vectors.pop(best_local_idx)
            
            print(f"Bulyan Phase 1 selected clients: {selected_indices}")
            
            # Phase 2: Coordinate-wise trimmed mean using L2 norms (same as TrimmedMean)
            print(f"Bulyan Phase 2: Computing trimmed mean on {len(selected_indices)} vectors with beta={beta}...")
            
            # Get L2 norms from client metrics for the selected clients
            selected_norms = [results[i][1].metrics.get("l2_norm", 0.0) for i in selected_indices]
            
            # Sort by norms and trim beta smallest and beta largest
            indexed_norms = sorted(enumerate(selected_norms), key=lambda x: x[1])
            
            if len(indexed_norms) <= 2 * beta:
                # Not enough vectors to trim, use all selected vectors
                indices_to_keep = list(range(len(selected_indices)))
                print(f"Bulyan Phase 2: Not enough vectors to trim (need >{2*beta}, have {len(selected_indices)}), using all selected vectors")
            else:
                # Trim beta from each end
                indices_to_keep = [i for i, _ in indexed_norms[beta:-beta]]
                print(f"Bulyan Phase 2: Trimmed {beta} vectors from each end, keeping {len(indices_to_keep)} vectors")
            
            # Map back to the original selected indices
            final_selected_indices = [selected_indices[i] for i in indices_to_keep]
            final_vectors = [client_vectors[i] for i in final_selected_indices]
            
            print(f"Bulyan final selected clients: {final_selected_indices}")
            
            # Safety check: if no vectors remain, fall back to simple average
            if not final_vectors:
                print("Warning: No vectors remaining after Bulyan, falling back to simple average")
                aggregated_vector = sum(client_vectors) * (1 / len(client_vectors))
            else:
                # Final aggregation
                aggregated_vector = sum(final_vectors) * (1 / len(final_vectors))
        
        dec_start = time.perf_counter()
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())
        dec_end = time.perf_counter()
        server_decryption_time = dec_end - dec_start

        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        
        return ndarrays_to_parameters(new_global_weights), {"server_decryption_time": server_decryption_time}


# ==============================================================================
# CKKS Metrics Strategy Wrapper
# ==============================================================================
class CKKSMetricsStrategyWrapper(Strategy):
    """A wrapper that sits on top of a CKKS strategy to collect and log metrics."""

    def __init__(self, base_strategy: Strategy, run_config: dict):
        super().__init__()
        self.base_strategy = base_strategy
        self.run_config = run_config
        self.run_name = run_config.get("run_name", "default_run")
        
        # State for tracking new metrics
        self.best_accuracy = 0.0
        self.convergence_round = -1 # -1 means not converged yet
        self.total_aggregation_time = 0.0
        self.total_client_encryption_time = 0.0  # CKKS-specific metric
        self.total_server_decryption_time = 0.0  # CKKS-specific metric
        self.total_client_training_time = 0.0 # NEW: Initialize total client training time
        self.all_round_data = [] # To store detailed data for artifact

    def initialize_parameters(self, client_manager):
        return self.base_strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],): # Type hints for clarity
        # --- 1. Time the aggregation step ---
        start_time = time.perf_counter()
        aggregated_params, agg_metrics = self.base_strategy.aggregate_fit(server_round, results, failures)
        end_time = time.perf_counter()
        aggregation_time = end_time - start_time
        self.total_aggregation_time += aggregation_time

        num_successful_clients = len(results)

        # --- 2. Calculate communication costs ---
        # Uplink: Sum of bytes from all successful client results
        total_uplink_kb = sum(res.metrics.get("uplink_kb", 0) for _, res in results)
        uplink_mb = total_uplink_kb / 1024 # Convert KB to MB

        # Downlink: Size of the newly aggregated global model
        downlink_mb = 0
        if aggregated_params:
            downlink_bytes = get_weights_size_bytes(parameters_to_ndarrays(aggregated_params))
            downlink_mb = downlink_bytes / (1024 * 1024)

        # --- 3. CKKS-specific metrics ---
        # Sum of client-reported encryption times
        per_round_client_encryption_time = sum(res.metrics.get("encryption_time", 0) for _, res in results)
        self.total_client_encryption_time += per_round_client_encryption_time
        
        # Server-side decryption time (passed from base strategy's agg_metrics)
        per_round_server_decryption_time = agg_metrics.get("server_decryption_time", 0)
        self.total_server_decryption_time += per_round_server_decryption_time

        # Average expansion factor across participating clients
        expansion_factors = [res.metrics.get("expansion_factor", 0) for _, res in results]
        avg_expansion_factor = np.mean(expansion_factors) if expansion_factors else 0

        # NEW: Extract and sum client training time
        # This will give the sum of training times across all participating clients in this round.
        per_round_client_training_time = sum(res.metrics.get("training_time", 0) for _, res in results)
        self.total_client_training_time += per_round_client_training_time


        # --- 4. Log per-round metrics to W&B ---
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
            # NEW: Add training time metrics
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
        print("Logging final CKKS metrics and artifacts...")

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
        file_path = os.path.join(run_dir, f"{self.run_name}_ckks_rundata.json")

        with open(file_path, "w") as f:
            json.dump(self.all_round_data, f, indent=2)

        artifact = wandb.Artifact(name=f"{self.run_name}_ckks_details", type="dataset")
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)

        print(f"✅ CKKS Artifact '{self.run_name}_ckks_details' logged in {run_dir}")


# ==============================================================================
# The main builder function for ALL CKKS strategies with metrics wrapper
# ==============================================================================
def build_strategy_ckks(run_config: dict) -> Tuple[Strategy, bytes]:
    strategy_name = run_config.get("strategy")
    num_malicious = run_config.get("num_malicious", 0)
    run_name = run_config.get("run_name", "default_ckks_run")

    wandb.init(project="fl-fhe-ckks-benchmark", name=run_name, reinit=True, settings=wandb.Settings(start_method="thread"))
    wandb.config.update(run_config)

    fhe_context = get_ckks_context()
    context_bytes = fhe_context.serialize(save_secret_key=False)

    # MultiKrum requires the smaller model for FHE dot products.
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
        config_to_send["fhe_context_bytes"] = context_bytes
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
    }
    
    # Create the base strategy
    base_strategy: Strategy
    if strategy_name == "CKKSKrum":
        base_strategy = CKKSKrum(num_malicious_clients=num_malicious, **common_args)
    elif strategy_name == "CKKSMultiKrum":
        base_strategy = CKKSMultiKrum(num_malicious_clients=num_malicious, num_clients_to_keep=run_config["num_partitions"] - num_malicious, **common_args)
    elif strategy_name == "CKKSTrimmedMean":
        base_strategy = CKKSTrimmedMean(num_malicious_clients=num_malicious, **common_args)
    elif strategy_name == "CKKSBulyan":
        selection_size = run_config.get("bulyan_selection_size")
        trimmed_mean_beta = run_config.get("trimmed_mean_beta")
        
        base_strategy = CKKSBulyan(
            num_malicious_clients=num_malicious,
            bulyan_selection_size=selection_size,
            trimmed_mean_beta=trimmed_mean_beta,
            **common_args
        )
    else: # Default to CKKSFedAvg
        base_strategy = CKKSFedAvg(**common_args)

    # Wrap the base strategy with metrics collection
    wrapped_strategy = CKKSMetricsStrategyWrapper(base_strategy=base_strategy, run_config=run_config)

    return wrapped_strategy, context_bytes
