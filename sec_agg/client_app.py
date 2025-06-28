import torch
from flwr.client import NumPyClient
from sec_agg.task import Net, load_data, get_weights, set_weights, train

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def client_fn(partition_id: int, num_partitions: int, run_config: dict):
    net = Net().to(DEVICE)
    alpha = run_config.get("alpha", 0.5)
    num_malicious = run_config.get("num_malicious", 0)
    is_byzantine = partition_id < num_malicious
    trainloader, _ = load_data(partition_id, num_partitions, alpha)

    class FlowerClient(NumPyClient):
        def fit(self, parameters, config):
            set_weights(net, parameters)
            if is_byzantine:
                print(f"[Client {partition_id}] Byzantine client - flipping weights")
                return [-w for w in get_weights(net)], len(trainloader.dataset), {}
            return train(net, trainloader, epochs=config["local_epochs"], device=DEVICE), len(trainloader.dataset), {}

    return FlowerClient().to_client()
