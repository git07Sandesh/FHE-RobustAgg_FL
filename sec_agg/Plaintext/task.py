# --- START OF FILE task.py ---

from collections import OrderedDict
from typing import Tuple, List
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from datasets import disable_caching
DATA_ROOT = os.environ.get("FL_DATA_ROOT", "./data")
TRANSFORMS = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ... (Net and SmallNet class definitions remain unchanged) ...
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.fc1 = nn.Linear(10 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 5)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ... (load_data, train, test, get_weights, set_weights, etc. remain unchanged) ...
def get_central_testloader() -> DataLoader:
    """Loads the central test set."""
    # [MODIFIED] Use the DATA_ROOT variable and set download=False.
    # The data must be pre-downloaded to this location.
    print(f"Loading central test set from: {DATA_ROOT}")
    testset = CIFAR10(root=DATA_ROOT, train=False, download=False, transform=TRANSFORMS)
    return DataLoader(testset, batch_size=64, shuffle=False)


# In sec_agg/Plaintext/task.py

def load_data(partition_id: int, num_partitions: int, alpha: float, batch_size: int = 64) -> Tuple[DataLoader, None]:
    """Loads a partition of the dataset for a single client using a robust manual method."""
    
    print(f"Client {partition_id}: Loading data partition manually from {DATA_ROOT}...")
    
    # Load the full CIFAR10 train set from the specified DATA_ROOT.
    # download=False is critical.
    try:
        dataset = CIFAR10(root=DATA_ROOT, train=True, download=False, transform=TRANSFORMS)
    except Exception as e:
        print(f"FATAL: Client {partition_id}: Failed to load CIFAR10 dataset from {DATA_ROOT}. Make sure it is pre-downloaded. Error: {e}")
        # Exit with an error code if data isn't found, so the job fails clearly.
        sys.exit(1)

    # The rest of your manual partitioning logic is great and remains unchanged.
    labels = np.array(dataset.targets)
    idx_by_class = [np.where(labels == i)[0] for i in range(len(dataset.classes))]

    # We use a fixed seed to ensure partitions are the same on every run
    rng = np.random.default_rng(12345)
    proportions = rng.dirichlet([alpha] * num_partitions, len(dataset.classes))

    partitions = [[] for _ in range(num_partitions)]
    for cls_idx, cls_prop in zip(idx_by_class, proportions):
        # Permute the indices for this class
        permuted_indices = rng.permutation(cls_idx)
        # Calculate the split points
        split_points = np.cumsum(cls_prop)[:-1] * len(cls_idx)
        # Split the permuted indices
        cls_splits = np.split(permuted_indices, split_points.astype(int))
        for pid, split in enumerate(cls_splits):
            partitions[pid].extend(split.tolist())

    indices = partitions[partition_id]
    subset = Subset(dataset, indices)
    
    print(f"Client {partition_id}: Successfully created manual partition with {len(indices)} samples.")
    return DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True), None

def train(net: nn.Module, trainloader: DataLoader, epochs: int, device) -> list:
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 5))

    net.train()

    for epoch in range(epochs):
        for batch in trainloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()  # Step once per local epoch (not per batch)

    return get_weights(net)
def train(net: nn.Module, trainloader: DataLoader, epochs: int, device) -> list:
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 1))

    net.train()

    for epoch in range(epochs):
        for batch in trainloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()  # Step once per local epoch (not per batch)

    return get_weights(net)

def test(net: nn.Module, testloader: DataLoader, device) -> Tuple[float, float]:
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    net.eval()
    correct, total, loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return loss / len(testloader), accuracy

def get_weights(net: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# [NEW] Helper function to calculate the size of model weights in bytes
def get_weights_size_bytes(weights: List[np.ndarray]) -> int:
    """Calculates the total size of a list of NumPy arrays in bytes."""
    return sum(w.nbytes for w in weights)

def flatten_weights(weights: list) -> np.ndarray:
    return np.concatenate([w.flatten() for w in weights])

# ... (unflatten_weights, get_trainloader remain unchanged) ...

def unflatten_weights(flat_weights: np.ndarray, model: nn.Module) -> list:
    """Reconstructs model weights from a flat array using the model’s state_dict."""
    new_weights = []
    offset = 0
    for param in model.parameters():
        shape = param.shape
        size = param.numel()
        param_flat = flat_weights[offset : offset + size]
        new_weights.append(param_flat.reshape(shape))
        offset += size
    return new_weights
def get_trainloader(partition_id: int, num_partitions: int = 5, alpha: float = 0.5, batch_size: int = 64):
    trainloader, _ = load_data(partition_id, num_partitions, alpha, batch_size)
    return trainloader
