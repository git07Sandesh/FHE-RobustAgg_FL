# sec_agg/server_app.py

import wandb
import torch
import tempfile
import os
import pickle
import time
import numpy as np
from typing import List, Tuple, Union, Optional, Dict

# FIXED: Import the 'aggregate' utility function directly
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
from sec_agg.Plaintext.task import Net, SmallNet, get_weights, get_central_testloader, set_weights, test, flatten_weights

# A small helper function that was previously part of Flower's public API
def fit_res_to_sample(fit_res: FitRes) -> Tuple[List[np.ndarray], int]:
    """Extracts weight parameters and number of examples from a FitRes."""
    parameters = parameters_to_ndarrays(fit_res.parameters)
    num_examples = fit_res.num_examples
    return parameters, num_examples

# ==============================================================================
# 1. Define Custom Strategy Implementations
# ==============================================================================

class MultiKrum(Krum):
    """
    An implementation of the Multi-Krum robust aggregation rule.
    It extends the basic Krum by selecting and averaging the top-k
    "best" client updates instead of just the single best one.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}
        
        num_clients = len(results)
        if num_clients < self.min_fit_clients: return None, {}
        
        weights = [parameters_to_ndarrays(res.parameters) for _, res in results]
        flat_weights = [flatten_weights(w) for w in weights]
        
        distances = []
        for i in range(num_clients):
            dists_i = []
            for j in range(num_clients):
                if i == j:
                    dists_i.append(float('inf'))
                else:
                    dist = np.linalg.norm(flat_weights[i] - flat_weights[j]) ** 2
                    dists_i.append(dist)
            distances.append(dists_i)

        m = num_clients - self.num_malicious_clients - 2
        scores = []
        for i in range(num_clients):
            sorted_dists = sorted(distances[i])
            if len(sorted_dists) > m :
                scores.append(sum(sorted_dists[:m]))
            else:
                scores.append(sum(sorted_dists))

        indexed_scores = list(zip(results, scores))
        sorted_clients = sorted(indexed_scores, key=lambda x: x[1])
        clients_to_keep = sorted_clients[:self.num_clients_to_keep]
        print(f"Multi-Krum selected {len(clients_to_keep)} clients.")
        
        # Extract the FitRes objects from the kept clients
        kept_fit_res = [res for (proxy, res), score in clients_to_keep]

        # Convert the kept results into the format needed for the aggregate function
        samples = [fit_res_to_sample(res) for res in kept_fit_res]
        
        # FIXED: Call the imported 'aggregate' utility function directly
        aggregated_ndarrays = aggregate(samples)
        
        # The 'aggregate' function returns ndarrays, so we need to convert them back to Parameters
        return ndarrays_to_parameters(aggregated_ndarrays), {}


class TrimmedMean(FedAvg):
    """
    An implementation of the Trimmed Mean robust aggregation rule.
    It discards a certain number of clients with the highest and lowest
    update norms before averaging the rest.
    """
    def __init__(self, num_malicious_clients: int, **kwargs):
        super().__init__(**kwargs)
        self.b = num_malicious_clients

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}
        
        samples = [fit_res_to_sample(res) for _, res in results]
        norms = [np.linalg.norm(flatten_weights(params)) for params, _ in samples]
        indexed_norms = sorted(enumerate(norms), key=lambda x: x[1])
        
        num_to_trim = self.b
        if len(indexed_norms) <= 2 * num_to_trim:
            print("Not enough clients to perform trimming, using all.")
            indices_to_keep = list(range(len(indexed_norms)))
        else:
            untrimmed_indices = [i for i, _ in indexed_norms]
            indices_to_keep = untrimmed_indices[num_to_trim : -num_to_trim]

        print(f"TrimmedMean trimmed {len(samples) - len(indices_to_keep)} clients, keeping {len(indices_to_keep)}.")
        
        samples_to_aggregate = [samples[i] for i in indices_to_keep]
        
        # FIXED: Call the imported 'aggregate' utility function directly
        aggregated_ndarrays = aggregate(samples_to_aggregate)

        return ndarrays_to_parameters(aggregated_ndarrays), {}

# ... (CoordinateWiseMedian and build_strategy are correct and remain the same) ...
class CoordinateWiseMedian(FedAvg):
    """
    An implementation of the Coordinate-wise Median robust aggregation rule.
    For each parameter in the model, it calculates the median value across
    all client updates.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results: return None, {}
            
        # Get all client weight updates
        client_updates = [parameters_to_ndarrays(res.parameters) for _, res in results]
        
        num_layers = len(client_updates[0])
        aggregated_weights = []
        
        for i in range(num_layers):
            # Stack the weights of the i-th layer from all clients
            layer_updates = np.stack([client[i] for client in client_updates])
            
            # Compute the median along the client axis (axis=0)
            median_layer = np.median(layer_updates, axis=0)
            aggregated_weights.append(median_layer)

        # The return type for aggregate_fit is (Parameters, Dict), so return an empty dict
        return ndarrays_to_parameters(aggregated_weights), {}


# ==============================================================================
# 2. The main builder function to create the strategy
# ==============================================================================
def build_strategy(run_config: dict) -> Strategy:
    run_name = run_config.get("run_name", "default_run")
    strategy_name = run_config.get("strategy", "FedAvg")
    num_rounds = run_config.get("num_rounds", 3)
    num_malicious = run_config.get("num_malicious", 0)
    local_epochs = run_config.get("local_epochs", 1)

    wandb.init(project="fl-baseline-research", name=run_name, reinit=True,
               settings=wandb.Settings(start_method="thread"))

    net = SmallNet()
    initial_parameters = ndarrays_to_parameters(get_weights(net))
    
    testloader = get_central_testloader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def evaluate(server_round: int, parameters: List[np.ndarray], _) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        set_weights(net, parameters)
        loss, acc = test(net, testloader, device)
        wandb.log({"round": server_round, "server_loss": loss, "server_accuracy": acc})
        
        if server_round == num_rounds:
             with tempfile.TemporaryDirectory() as tmpdir:
                weights_path = os.path.join(tmpdir, f"{run_name}_final_weights.pkl")
                with open(weights_path, "wb") as f:
                    pickle.dump(parameters, f)
                artifact = wandb.Artifact(name=f"{run_name}_weights", type="model")
                artifact.add_file(weights_path)
                wandb.log_artifact(artifact)
        
        return loss, {"accuracy": acc}

    def fit_config(server_round):
        return {"local_epochs": local_epochs}
    
    strategy: Strategy
    common_args = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 0.0,
        "min_available_clients": run_config["num_partitions"],
        "evaluate_fn": evaluate,
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
    elif strategy_name == "CoordMedian":
        base_strategy = CoordinateWiseMedian(**common_args)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
        
    wandb.config.update(run_config)
    wandb.config.update({"strategy": strategy_name})

    return base_strategy