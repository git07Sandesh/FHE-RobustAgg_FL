# sec_agg/server_he.py

import wandb
import torch
import time
import tenseal as ts
import numpy as np

from typing import List, Tuple, Union, Optional, Dict
from flwr.server.strategy import FedAvg
from flwr.server.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from .task import Net, SmallNet, get_central_testloader, get_weights, set_weights, test, unflatten_weights
from .task_he import get_fhe_context

# ==============================================================================
# Define the HE-enabled FedAvg Strategy
# ==============================================================================

class HEFedAvg(FedAvg):
    def __init__(self, fhe_context: ts.Context, **kwargs):
        self.fhe_context = fhe_context
        self.sample_model = Net()
        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        num_examples_total = 0
        weighted_vectors = []
        aggregation_start_time = time.time() # Start timer
        for _, fit_res in results:
            uint8_array = parameters_to_ndarrays(fit_res.parameters)[0]
            encrypted_vector_bytes = uint8_array.tobytes()
            encrypted_vector = ts.ckks_vector_from(self.fhe_context, encrypted_vector_bytes)

            num_examples = fit_res.num_examples
            encrypted_vector *= num_examples  # In-place multiplication by scalar is supported
            
            weighted_vectors.append(encrypted_vector)
            num_examples_total += num_examples
            
        if not weighted_vectors:
            return None, {}

        print("Homomorphically aggregating client updates...")
        aggregated_vector = sum(weighted_vectors)
        
        if num_examples_total == 0:
            return None, {}

        # FINAL FIX: Multiply by the inverse of the total number of examples.
        # This is the correct way to perform division by a plaintext scalar in TenSEAL.
        inverse_total = 1 / num_examples_total
        aggregated_vector *= inverse_total
        aggregation_duration = time.time() - aggregation_start_time
        print(f"Aggregation took {aggregation_duration:.4f}s")
        wandb.log({"server_aggregation_time": aggregation_duration})

        print("Decrypting aggregated result...")
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())

        new_global_weights_ndarrays = unflatten_weights(decrypted_flat_weights, self.sample_model)

        return ndarrays_to_parameters(new_global_weights_ndarrays), {}
    
# ==============================================================================
# Define the HE-enabled Krum Strategy
# ==============================================================================
class HEKrum(FedAvg):
    def __init__(self, num_malicious_clients: int, num_clients_to_keep: int, fhe_context: ts.Context, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious_clients = num_malicious_clients
        self.num_clients_to_keep = num_clients_to_keep
        self.fhe_context = fhe_context
        # Use the smaller model for Krum
        self.sample_model = SmallNet() 

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # 1. Deserialize all client updates into CKKS vectors
        client_vectors = []
        aggregation_start_time = time.time() # Start timer
        for _, fit_res in results:
            uint8_array = parameters_to_ndarrays(fit_res.parameters)[0]
            encrypted_vector_bytes = uint8_array.tobytes()
            client_vectors.append(ts.ckks_vector_from(self.fhe_context, encrypted_vector_bytes))
        
        n_clients = len(client_vectors)
        
        # 2. Homomorphically compute pairwise squared distances
        print("Homomorphically computing pairwise squared distances...")
        scores = []
        for i in range(n_clients):
            distances = []
            for j in range(n_clients):
                if i == j:
                    continue
                # distance_vec = u_i - u_j
                distance_vec = client_vectors[i] - client_vectors[j]
                # squared_distance = (u_i - u_j) . (u_i - u_j)
                # This is the dot product that requires a single ciphertext!
                squared_distance = distance_vec.dot(distance_vec) 
                distances.append(squared_distance)
            
            # 3. Sort distances and sum the k-f-2 smallest ones
            # THIS IS A SIMPLIFICATION. A true FHE implementation would need a
            # secure sorting network. For now, we are skipping this complex step
            # and just summing all distances. This is a common simplification.
            # A more advanced version could use a secure top-k protocol.
            scores.append(sum(distances))

        aggregation_duration = time.time() - aggregation_start_time
        print(f"Aggregation took {aggregation_duration:.4f}s")
        wandb.log({"server_aggregation_time": aggregation_duration})
        
        # 4. Decrypt ONLY the final scores
        print("Decrypting Krum scores...")
        decrypted_scores = [score.decrypt()[0] for score in scores] # Decrypt and get the single scalar value
        
        # 5. Select the best clients based on decrypted scores
        # Associate scores with their original indices
        indexed_scores = sorted(enumerate(decrypted_scores), key=lambda x: x[1])
        
        # Get the indices of the clients to keep
        indices_to_keep = [idx for idx, score in indexed_scores[:self.num_clients_to_keep]]
        print(f"Krum selected clients with indices: {indices_to_keep}")
        
        # 6. Homomorphically average the updates from the selected clients
        selected_updates = [client_vectors[i] for i in indices_to_keep]
        aggregated_vector = sum(selected_updates) * (1 / len(selected_updates))
        
        # 7. Decrypt the final model and return
        decrypted_flat_weights = np.array(aggregated_vector.decrypt())
        new_global_weights = unflatten_weights(decrypted_flat_weights, self.sample_model)
        
        return ndarrays_to_parameters(new_global_weights), {}


# ==============================================================================
# The main builder function for HE strategies (FedAvg or Krum)
# ==============================================================================
def build_strategy_he(run_config: dict) -> Tuple[FedAvg, bytes]:
    run_name = run_config.get("run_name", "default_run")
    strategy_name = run_config.get("strategy", "HEFedAvg") # Change it in toml file while switching to Krum
    num_rounds = run_config.get("num_rounds", 3)
    local_epochs = run_config.get("local_epochs", 1)
    num_malicious = run_config.get("num_malicious", 0)

    wandb.init(project="fl-fhe-research", name=run_name, reinit=True,
               settings=wandb.Settings(start_method="thread"))
    wandb.config.update(run_config)

    # Setup FHE context
    fhe_context = get_fhe_context()
    context_bytes = fhe_context.serialize(save_secret_key=False)

    # Pick model based on strategy
    if strategy_name == "HEKrum":
        model_to_use = SmallNet()
    else: # Default to HEFedAvg
        model_to_use = Net()
    print(f"✅ Server using model: {model_to_use.__class__.__name__} for run: {run_name}")

    initial_parameters = ndarrays_to_parameters(get_weights(model_to_use))

    # Evaluation and Config Functions
    testloader = get_central_testloader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round: int, parameters: List[np.ndarray], _):
        set_weights(model_to_use, parameters)
        loss, acc = test(model_to_use, testloader, device)
        wandb.log({"round": server_round, "server_loss": loss, "server_accuracy": acc})
        return loss, {"accuracy": acc}

    def fit_config(server_round: int):
        return {
            "local_epochs": local_epochs,
            "fhe_context_bytes": context_bytes,
        }

    # Strategy selection
    if strategy_name == "HEKrum":
        strategy = HEKrum(
            fhe_context=fhe_context,
            num_malicious_clients=num_malicious,
            num_clients_to_keep=run_config["num_partitions"] - num_malicious,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_available_clients=run_config["num_partitions"],
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate,
            on_fit_config_fn=fit_config,
        )
    else:
        strategy = HEFedAvg(
            fhe_context=fhe_context,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_available_clients=run_config["num_partitions"],
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate,
            on_fit_config_fn=fit_config,
        )

    return strategy, context_bytes
