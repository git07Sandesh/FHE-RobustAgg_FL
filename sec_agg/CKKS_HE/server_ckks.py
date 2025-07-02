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
        
        decrypted_flat_weights = np.array(selected_vector.decrypt())
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
            
            selected_vectors = [client_vectors[i] for i in selected_indices]
            
            print(f"Bulyan Phase 1 selected clients: {selected_indices}")
            
            # Phase 2: Coordinate-wise trimmed mean
            print(f"Bulyan Phase 2: Computing coordinate-wise trimmed mean on {len(selected_vectors)} vectors with beta={beta}...")
            
            # For CKKS, we need to work with the encrypted vectors directly
            # We'll compute a "pseudo" trimmed mean by removing extreme values
            
            # Since we can't do coordinate-wise operations easily in CKKS,
            # we'll use a simplified approach: trim based on vector norms
            if len(selected_vectors) > 2 * beta:
                # Compute norms of selected vectors homomorphically
                norms = []
                for vec in selected_vectors:
                    norm_squared = vec.dot(vec)
                    norms.append(norm_squared)
                
                print("Decrypting norms for trimming...")
                decrypted_norms = [norm.decrypt()[0] for norm in norms]
                
                # Sort by norms and trim beta smallest and beta largest
                indexed_norms = sorted(enumerate(decrypted_norms), key=lambda x: x[1])
                
                # Keep the middle vectors (trim beta from each end)
                trimmed_indices = [i for i, _ in indexed_norms[beta:-beta]] if len(indexed_norms) > 2 * beta else list(range(len(indexed_norms)))
                final_vectors = [selected_vectors[i] for i in trimmed_indices]
                
                print(f"Bulyan Phase 2 kept {len(final_vectors)} vectors after trimming {beta} from each end")
            else:
                final_vectors = selected_vectors
                print(f"Bulyan Phase 2: Not enough vectors to trim (need >{2*beta}, have {len(selected_vectors)}), using all selected vectors")
            
            # Final aggregation
            if final_vectors:
                aggregated_vector = sum(final_vectors) * (1 / len(final_vectors))
            else:
                print("Warning: No vectors remaining after Bulyan, falling back to simple average")
                aggregated_vector = sum(client_vectors) * (1 / len(client_vectors))
        
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())
        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        
        return ndarrays_to_parameters(new_global_weights), {}


# Enhanced version with better coordinate-wise trimming approximation
# class CKKSBulyanEnhanced(FedAvg):
#     def __init__(self, num_malicious_clients: int, fhe_context: ts.Context, sample_model: torch.nn.Module,
#                  bulyan_selection_size: Optional[int] = None, trimmed_mean_beta: Optional[int] = None, **kwargs):
#         super().__init__(**kwargs)
#         self.num_malicious_clients = num_malicious_clients
#         self.fhe_context = fhe_context
#         self.sample_model = sample_model
#         self.bulyan_selection_size = bulyan_selection_size
#         self.trimmed_mean_beta = trimmed_mean_beta if trimmed_mean_beta is not None else num_malicious_clients

#     def aggregate_fit(
#         self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
#     ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
#         if not results: return None, {}

#         client_vectors = [ts.ckks_vector_from(self.fhe_context, parameters_to_ndarrays(res.parameters)[0].tobytes()) for _, res in results]
#         n_clients = len(client_vectors)
#         f = self.num_malicious_clients
        
#         # Bulyan parameters
#         theta = self.bulyan_selection_size if self.bulyan_selection_size is not None else n_clients - 2 * f
#         beta = self.trimmed_mean_beta
        
#         print(f"Bulyan Enhanced parameters: theta={theta}, beta={beta}, n_clients={n_clients}, f={f}")
        
#         if theta <= 0:
#             print(f"Warning: Bulyan requires n > 2f, but got n={n_clients}, f={f}")
#             aggregated_vector = sum(client_vectors) * (1 / len(client_vectors))
#         else:
#             print(f"Bulyan Enhanced Phase 1: Iterative Krum selection of {theta} clients...")
            
#             # Phase 1: Iterative Krum selection (same as standard version)
#             remaining_indices = list(range(n_clients))
#             remaining_vectors = client_vectors.copy()
#             selected_indices = []
            
#             for selection_round in range(theta):
#                 if len(remaining_vectors) <= 1:
#                     break
                    
#                 print(f"Krum selection round {selection_round + 1}/{theta}")
                
#                 # Compute Krum scores for remaining clients
#                 scores = []
#                 for i in range(len(remaining_vectors)):
#                     distances = []
#                     for j in range(len(remaining_vectors)):
#                         if i == j: continue
#                         distance_vec = remaining_vectors[i] - remaining_vectors[j]
#                         squared_distance = distance_vec.dot(distance_vec)
#                         distances.append(squared_distance)
#                     scores.append(sum(distances))
                
#                 # Decrypt scores and find the best client
#                 decrypted_scores = [score.decrypt()[0] for score in scores]
#                 best_local_idx = np.argmin(decrypted_scores)
#                 best_global_idx = remaining_indices[best_local_idx]
                
#                 # Select this client and remove from remaining
#                 selected_indices.append(best_global_idx)
#                 remaining_indices.pop(best_local_idx)
#                 remaining_vectors.pop(best_local_idx)
            
#             selected_vectors = [client_vectors[i] for i in selected_indices]
            
#             print(f"Selected clients: {selected_indices}")
            
#             # Phase 2: Enhanced trimming using multiple criteria
#             print(f"Bulyan Enhanced Phase 2: Multi-criteria trimming with beta={beta}...")
            
#             if len(selected_vectors) > 2 * beta:
#                 # Compute multiple statistics for each vector
#                 vector_stats = []
#                 for i, vec in enumerate(selected_vectors):
#                     # L2 norm
#                     norm_squared = vec.dot(vec)
                    
#                     # Average pairwise distance to other selected vectors
#                     avg_distance = 0
#                     for j, other_vec in enumerate(selected_vectors):
#                         if i != j:
#                             diff = vec - other_vec
#                             distance = diff.dot(diff)
#                             avg_distance = avg_distance + distance
#                     avg_distance = avg_distance * (1 / (len(selected_vectors) - 1))
                    
#                     vector_stats.append((i, norm_squared, avg_distance))
                
#                 print("Decrypting statistics for enhanced trimming...")
#                 # Decrypt all statistics
#                 decrypted_stats = []
#                 for i, norm_sq, avg_dist in vector_stats:
#                     norm_val = norm_sq.decrypt()[0]
#                     dist_val = avg_dist.decrypt()[0]
#                     decrypted_stats.append((i, norm_val, dist_val))
                
#                 # Combined scoring: normalize and combine norm and distance metrics
#                 combined_scores = []
#                 norms = [norm for _, norm, _ in decrypted_stats]
#                 dists = [dist for _, _, dist in decrypted_stats]
                
#                 norm_mean, norm_std = np.mean(norms), np.std(norms) + 1e-8
#                 dist_mean, dist_std = np.mean(dists), np.std(dists) + 1e-8
                
#                 for i, norm, dist in decrypted_stats:
#                     norm_score = abs(norm - norm_mean) / norm_std
#                     dist_score = abs(dist - dist_mean) / dist_std
#                     combined_score = norm_score + dist_score  # Higher score = more outlier-like
#                     combined_scores.append((i, combined_score))
                
#                 # Sort by combined score and keep the middle ones
#                 combined_scores.sort(key=lambda x: x[1])
                
#                 # Trim beta vectors from each end based on combined outlier score
#                 num_to_keep = len(combined_scores) - 2 * beta
#                 if num_to_keep > 0:
#                     kept_indices = [idx for idx, _ in combined_scores[:num_to_keep]]
#                 else:
#                     kept_indices = [idx for idx, _ in combined_scores]
                
#                 final_vectors = [selected_vectors[i] for i in kept_indices]
#                 print(f"Enhanced trimming kept {len(final_vectors)} vectors (trimmed {beta} from each end)")
#             else:
#                 final_vectors = selected_vectors
#                 print(f"Not enough vectors to trim, using all {len(final_vectors)}")
            
#             # Final aggregation
#             if final_vectors:
#                 aggregated_vector = sum(final_vectors) * (1 / len(final_vectors))
#             else:
#                 aggregated_vector = sum(client_vectors) * (1 / len(client_vectors))
        
#         decrypted_flat_weights = np.array(aggregated_vector.decrypt())
#         new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        
#         return ndarrays_to_parameters(new_global_weights), {}

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
    model_to_use = SmallNet()
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
    if strategy_name == "CKKSKrum":
        strategy = CKKSKrum(num_malicious_clients=num_malicious, **common_args)
    elif strategy_name == "CKKSMultiKrum":
        strategy = CKKSMultiKrum(num_malicious_clients=num_malicious, num_clients_to_keep=run_config["num_partitions"] - num_malicious, **common_args)
    elif strategy_name == "CKKSTrimmedMean":
        strategy = CKKSTrimmedMean(num_malicious_clients=num_malicious, **common_args)
    elif strategy_name == "CKKSBulyan":
        selection_size = run_config.get("bulyan_selection_size")
        trimmed_mean_beta = run_config.get("trimmed_mean_beta")
        
        strategy = CKKSBulyan(
            num_malicious_clients=num_malicious,
            bulyan_selection_size=selection_size,      # Pass it here
            trimmed_mean_beta=trimmed_mean_beta,      # And pass it here
            **common_args
        )
    else: # Default to CKKSFedAvg
        strategy = CKKSFedAvg(**common_args)

    return strategy, context_bytes