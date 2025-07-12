# --- START OF FILE task.py ---

from collections import OrderedDict
from typing import Tuple, List
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from datasets import disable_caching

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
        self.fc1 = nn.Linear(10 * 5 * 5, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 5)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ... (load_data, train, test, get_weights, set_weights, etc. remain unchanged) ...

def get_central_testloader() -> DataLoader:
    testset = CIFAR10(root="./data", train=False, download=True, transform=TRANSFORMS)
    return DataLoader(testset, batch_size=64, shuffle=False)

def load_data(partition_id: int, num_partitions: int, alpha: float, batch_size: int = 32) -> Tuple[DataLoader, None]:
    try:
        # Try to use flwr_datasets (assumes internet access)
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions, partition_by="label", alpha=alpha,
            min_partition_size=100, seed=42,
        )
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
        partition = fds.load_partition(partition_id, "train")

        def apply_transforms(batch):
            batch["img"] = [TRANSFORMS(img) for img in batch["img"]]
            return batch

        partition = partition.with_transform(apply_transforms)
        return DataLoader(partition, batch_size=batch_size, shuffle=True, drop_last=True), None

    except Exception as e:
        print("[WARNING] FederatedDataset failed, falling back to local CIFAR-10")
        from torchvision.datasets import CIFAR10
        from torch.utils.data import Subset

        # Load the full CIFAR10 train set locally
        dataset = CIFAR10(root="./data", train=True, download=False, transform=TRANSFORMS)

        # Create Dirichlet partitions manually (once only)
        labels = np.array(dataset.targets)
        num_classes = 10
        idx_by_class = [np.where(labels == i)[0] for i in range(num_classes)]

        # Sample proportions for each class
        class_counts = [len(idx) for idx in idx_by_class]
        proportions = np.random.dirichlet([alpha] * num_partitions, num_classes)

        # Build partition index list
        partitions = [[] for _ in range(num_partitions)]
        for cls_idx, cls_prop in zip(idx_by_class, proportions):
            cls_splits = np.split(np.random.permutation(cls_idx), 
                                  (np.cumsum(cls_prop)[:-1] * len(cls_idx)).astype(int))
            for pid, split in enumerate(cls_splits):
                partitions[pid].extend(split.tolist())

        indices = partitions[partition_id]
        subset = Subset(dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True), None
def train(net: nn.Module, trainloader: DataLoader, epochs: int, device) -> list:
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    net.train()

    for epoch in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
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
def get_trainloader(partition_id: int, num_partitions: int = 5, alpha: float = 0.5, batch_size: int = 32):
    trainloader, _ = load_data(partition_id, num_partitions, alpha, batch_size)
    return trainloader