# sec_agg/DP/client_dp.py

import torch
import numpy as np
from opacus import PrivacyEngine
from flwr.client import NumPyClient
# Ensure you have a get_trainloader in your task file that can set batch_size
from sec_agg.Plaintext.task import Net, SmallNet, get_trainloader, set_weights, get_weights

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# A new DP-specific train function
def train_dp(net, trainloader, run_config, device):
    """
    Trains a model with Differential Privacy using Opacus.
    A new PrivacyEngine is created for each call to ensure clean state.
    """
    # Get DP parameters from the config
    target_epsilon = run_config.get("dp_epsilon", 1.0)
    max_grad_norm = run_config.get("dp_max_grad_norm", 1.0)
    epochs = run_config.get("local_epochs", 1)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    # --- Opacus Integration ---
    # Create a new PrivacyEngine for this training run
    privacy_engine = PrivacyEngine()
    
    # Attach the privacy engine
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=net,
        optimizer=optimizer,
        data_loader=trainloader,
        target_epsilon=target_epsilon,
        target_delta=1e-5,  # A common value for CIFAR-10
        epochs=epochs,
        max_grad_norm=max_grad_norm,
    )
    # -------------------------

    model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(data_loader):
            images, labels = batch["img"].to(device), batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Get the privacy budget spent
    epsilon_spent = privacy_engine.get_epsilon(delta=1e-5)
    print(f"[Client] Epsilon spent: {epsilon_spent:.2f} (Target was {target_epsilon})")
    
    # No need to call detach or remove_hooks. The engine will be garbage collected.
    
    # Opacus modifies the model in-place, so we can just get its weights
    return get_weights(model)

class DPFlowerClient(NumPyClient):
    def __init__(self, net, trainloader, run_config):
        self.net = net
        self.trainloader = trainloader
        self.run_config = run_config

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        
        # Call the new DP training function
        trained_weights = train_dp(self.net, self.trainloader, self.run_config, DEVICE)
        
        partition_id = self.run_config.get("partition_id", -1)
        if partition_id < self.run_config.get("num_malicious", 0):
             trained_weights = [-w for w in trained_weights]
        
        return trained_weights, len(self.trainloader.dataset), {}

def client_fn_dp(partition_id: int, run_config: dict):
    strategy = run_config.get("strategy")
    net = SmallNet().to(DEVICE)
  
    
    # Ensure get_trainloader is imported and can handle batch_size
    trainloader = get_trainloader(
        partition_id=partition_id,
        num_partitions=run_config["num_partitions"],
        alpha=run_config["alpha"],
        batch_size=32 # A fixed batch size is important for DP calculations
    )
    
    client_run_config = run_config.copy()
    client_run_config["partition_id"] = partition_id
    
    return DPFlowerClient(net, trainloader, client_run_config).to_client()