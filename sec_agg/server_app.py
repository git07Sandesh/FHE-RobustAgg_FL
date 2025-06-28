import wandb
import torch
import tempfile
import os
import pickle
from flwr.server.strategy import FedAvg, Krum
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from sec_agg.task import Net, get_weights, get_central_testloader, set_weights, test

def build_strategy(run_config: dict):
    run_name = run_config.get("run_name", "default_run")
    strategy_name = run_config.get("strategy", "FedAvg")
    num_rounds = run_config.get("num_rounds", 3)
    fraction_fit = run_config.get("fraction_fit", 1.0)
    num_malicious = run_config.get("num_malicious", 0)
    local_epochs = run_config.get("local_epochs", 1)

    wandb.init(project="fl-baseline-research", name=run_name, reinit=True)

    net = Net()
    initial_parameters = ndarrays_to_parameters(get_weights(net))
    testloader = get_central_testloader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round, parameters, _):
        set_weights(net, parameters)
        loss, acc = test(net, testloader, device)

        # Log metrics to wandb
        wandb.log({"round": server_round, "server_loss": loss, "server_accuracy": acc})

        # Save final weights to WandB as artifact at final round
        if server_round == num_rounds:
            final_weights = parameters  # Already a list of ndarrays

            with tempfile.TemporaryDirectory() as tmpdir:
                weights_path = os.path.join(tmpdir, f"{run_name}_final_weights.pkl")
                with open(weights_path, "wb") as f:
                    pickle.dump(final_weights, f)

                artifact = wandb.Artifact(name=f"{run_name}_weights", type="model")
                artifact.add_file(weights_path)
                wandb.log_artifact(artifact)

            print(f"✅ Final weights logged to WandB: {run_name}_weights")

        return loss, {"accuracy": acc}


    def fit_config(server_round):
        return {"local_epochs": local_epochs}

    kwargs = {
        "fraction_fit": fraction_fit,
        "fraction_evaluate": 0.0,
        "min_available_clients": run_config["num_partitions"],
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate,
        "on_fit_config_fn": fit_config,
    }

    if strategy_name == "Krum":
        kwargs["num_malicious_clients"] = num_malicious
        kwargs["num_clients_to_keep"] = run_config["num_partitions"] - num_malicious

    wandb.config.update(run_config)
    wandb.config.update({"strategy": strategy_name})

    strategy_map = {
        "FedAvg": FedAvg,
        "Krum": Krum,
    }

    return strategy_map[strategy_name](**kwargs)
