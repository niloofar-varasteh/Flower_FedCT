"""
ML Task: Model, Data, Training for FedCT with IID/Non-IID support
Contains all ML-related code: model definition, data loading, training, evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import numpy as np
from typing import List, Tuple
from collections import defaultdict


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for image classification"""

    def __init__(self, num_classes=10, in_channels=3, img_size=32):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Calculate size after 3 pooling layers
        final_size = img_size // (2 ** 3)  # 3 pooling layers
        self.fc1 = nn.Linear(256 * final_size * final_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


def get_model(dataset: str = "CIFAR10"):
    """Get model instance based on dataset

    Args:
        dataset: Dataset name - "CIFAR10" or "FashionMNIST"

    Returns:
        Model instance
    """
    if dataset == "CIFAR10":
        return SimpleCNN(num_classes=10, in_channels=3, img_size=32)
    elif dataset == "FashionMNIST":
        return SimpleCNN(num_classes=10, in_channels=1, img_size=28)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


# ============================================================================
# DATASET CLASSES
# ============================================================================

class UnlabeledDataset(Dataset):
    """Public unlabeled dataset (no labels available)"""

    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        image, _ = self.base_dataset[base_idx]  # Ignore label - truly unlabeled
        return image


class PseudoLabeledDataset(Dataset):
    """Dataset with pseudo-labels from consensus"""

    def __init__(self, images: List[torch.Tensor], pseudo_labels: List[int]):
        self.images = images
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.pseudo_labels[idx]


# ============================================================================
# DATA PARTITIONING STRATEGIES
# ============================================================================

def partition_data_iid(private_indices, num_partitions, seed=42):
    """
    IID partitioning: Randomly distribute data equally among clients

    Args:
        private_indices: Indices to partition
        num_partitions: Number of clients
        seed: Random seed for reproducibility

    Returns:
        List of index arrays, one per client
    """
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(private_indices)

    samples_per_client = len(shuffled_indices) // num_partitions
    partitions = []

    for i in range(num_partitions):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        partitions.append(shuffled_indices[start_idx:end_idx])

    return partitions


def partition_data_non_iid_dirichlet(trainset, private_indices, num_partitions, num_classes=10, alpha=0.5, seed=42):
    """
    Non-IID partitioning using Dirichlet distribution

    Lower alpha = more non-IID (e.g., 0.1 = highly non-IID, 1.0 = moderately non-IID)

    Args:
        trainset: Original training dataset (to access labels)
        private_indices: Indices to partition
        num_partitions: Number of clients
        num_classes: Number of classes in dataset
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed

    Returns:
        List of index arrays, one per client
    """
    np.random.seed(seed)

    # Get labels for private indices
    if hasattr(trainset, 'targets'):
        all_labels = np.array(trainset.targets)
    elif hasattr(trainset, 'labels'):
        all_labels = np.array(trainset.labels)
    else:
        # Fallback: manually extract labels
        all_labels = np.array([trainset[i][1] for i in range(len(trainset))])

    private_labels = all_labels[private_indices]

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in zip(private_indices, private_labels):
        class_indices[label].append(idx)

    # Sample proportions from Dirichlet distribution for each class
    partitions = [[] for _ in range(num_partitions)]

    for class_id in range(num_classes):
        indices = np.array(class_indices[class_id])
        np.random.shuffle(indices)

        # Sample proportions for this class
        proportions = np.random.dirichlet([alpha] * num_partitions)
        proportions = (proportions * len(indices)).astype(int)

        # Adjust to ensure all samples are distributed
        proportions[-1] = len(indices) - proportions[:-1].sum()

        # Distribute indices according to proportions
        start_idx = 0
        for client_id, count in enumerate(proportions):
            end_idx = start_idx + count
            partitions[client_id].extend(indices[start_idx:end_idx])
            start_idx = end_idx

    # Shuffle each partition
    for i in range(num_partitions):
        np.random.shuffle(partitions[i])
        partitions[i] = np.array(partitions[i])

    return partitions


def partition_data_non_iid_shards(trainset, private_indices, num_partitions, shards_per_client=2, seed=42):
    """
    Non-IID partitioning using class shards (each client gets few classes)

    Args:
        trainset: Original training dataset
        private_indices: Indices to partition
        num_partitions: Number of clients
        shards_per_client: Number of class shards per client
        seed: Random seed

    Returns:
        List of index arrays, one per client
    """
    np.random.seed(seed)

    # Get labels
    if hasattr(trainset, 'targets'):
        all_labels = np.array(trainset.targets)
    elif hasattr(trainset, 'labels'):
        all_labels = np.array(trainset.labels)
    else:
        all_labels = np.array([trainset[i][1] for i in range(len(trainset))])

    private_labels = all_labels[private_indices]

    # Sort indices by label
    sorted_indices = private_indices[np.argsort(private_labels)]

    # Divide into shards
    num_shards = num_partitions * shards_per_client
    shard_size = len(sorted_indices) // num_shards
    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]

    # Randomly assign shards to clients
    shard_indices = list(range(num_shards))
    np.random.shuffle(shard_indices)

    partitions = [[] for _ in range(num_partitions)]
    for client_id in range(num_partitions):
        client_shards = shard_indices[client_id * shards_per_client:(client_id + 1) * shards_per_client]
        for shard_id in client_shards:
            partitions[client_id].extend(shards[shard_id])
        partitions[client_id] = np.array(partitions[client_id])

    return partitions


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(partition_id: int, num_partitions: int = 5, unlabeled_size: int = 500, batch_size: int = 32,
              dataset: str = "CIFAR10", data_distribution: str = "iid", alpha: float = 0.5,
              shards_per_client: int = 2):
    """
    Load dataset for a specific partition with IID or Non-IID distribution

    Args:
        partition_id: ID of this client (0 to num_partitions-1)
        num_partitions: Total number of clients
        unlabeled_size: Size of public unlabeled dataset U (shared across all clients)
        batch_size: Batch size for training
        dataset: Dataset name - "CIFAR10" or "FashionMNIST"
        data_distribution: "iid", "non-iid-dirichlet", or "non-iid-shards"
        alpha: Dirichlet concentration parameter (for non-iid-dirichlet)
        shards_per_client: Number of shards per client (for non-iid-shards)

    Returns:
        trainloader: DataLoader for private training data
        valloader: DataLoader for test/validation data
        public_dataset: Unlabeled public dataset U
    """

    # Define transforms based on dataset
    if dataset == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        num_classes = 10
    elif dataset == "FashionMNIST":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform_test
        )
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported: CIFAR10, FashionMNIST")

    # Split: public unlabeled dataset + private data
    np.random.seed(42)  # For reproducibility
    all_indices = np.arange(len(trainset))

    # Sample indices for public dataset U
    public_indices = np.random.choice(all_indices, size=unlabeled_size, replace=False)

    # Remaining indices for private data
    private_mask = np.ones(len(trainset), dtype=bool)
    private_mask[public_indices] = False
    private_indices = all_indices[private_mask]

    # Create public unlabeled dataset
    public_dataset = UnlabeledDataset(trainset, public_indices)

    # Partition private data based on distribution type
    dist_lower = data_distribution.lower()
    if dist_lower == "iid":
        partitions = partition_data_iid(private_indices, num_partitions)
        dist_info = "IID"
    elif dist_lower == "non-iid-dirichlet":
        alpha = float(alpha) if alpha is not None else 0.5
        partitions = partition_data_non_iid_dirichlet(
            trainset, private_indices, num_partitions, num_classes, alpha
        )
        dist_info = f"Non-IID (Dirichlet α={alpha})"
    elif dist_lower == "non-iid-shards":
        partitions = partition_data_non_iid_shards(
            trainset, private_indices, num_partitions, shards_per_client
        )
        dist_info = f"Non-IID (Shards={shards_per_client}/client)"
    else:
        raise ValueError(f"Unsupported distribution: {data_distribution}")

    # Get this client's partition
    client_indices = partitions[partition_id]

    # Analyze class distribution for this client
    if hasattr(trainset, 'targets'):
        all_labels = np.array(trainset.targets)
    elif hasattr(trainset, 'labels'):
        all_labels = np.array(trainset.labels)
    else:
        all_labels = np.array([trainset[i][1] for i in range(len(trainset))])

    client_labels = all_labels[client_indices]
    class_counts = np.bincount(client_labels, minlength=num_classes)

    # Create dataloaders
    # Note: num_workers=0 for Ray compatibility in simulation mode
    # Create dataloaders
    # Note: num_workers=0 for Ray compatibility in simulation mode
    client_trainset = Subset(trainset, client_indices)
    trainloader = DataLoader(client_trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    # Analyze class distribution for this client
    if hasattr(trainset, 'targets'):
        all_labels = np.array(trainset.targets)
    elif hasattr(trainset, 'labels'):
        all_labels = np.array(trainset.labels)
    else:
        all_labels = np.array([trainset[i][1] for i in range(len(trainset))])

    client_labels = all_labels[client_indices]
    class_counts = np.bincount(client_labels, minlength=num_classes)

    # Print detailed data statistics with distribution info
    print(f"\n{'=' * 60}")
    print(f"✓ Client {partition_id + 1} Data Summary:")
    print(f"{'=' * 60}")
    print(f"  Dataset: {dataset}")
    print(f"  Distribution: {dist_info}")  # ← CHANGED from "IID (default)"
    print(f"  Private training samples: {len(client_trainset)}")
    print(f"  Public unlabeled samples (shared): {unlabeled_size}")
    print(f"  Test samples: {len(testset)}")
    print(f"\n  Class Distribution:")
    for class_id, count in enumerate(class_counts):
        percentage = (count / len(client_labels)) * 100 if len(client_labels) > 0 else 0
        print(f"    Class {class_id}: {count:4d} samples ({percentage:5.1f}%)")
    print(f"{'=' * 60}\n")

    return trainloader, valloader, public_dataset
# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train(net, trainloader, epochs: int, device: torch.device, optimizer_name: str = "SGD", learning_rate: float = 0.01,
          public_loader=None):
    """
    Train the network with mixed batch sampling

    Args:
        net: Neural network model
        trainloader: DataLoader with private training data
        epochs: Number of epochs (full passes through the dataset)
        device: Device to train on (cuda/cpu)
        optimizer_name: Optimizer to use - "SGD" or "Adam"
        learning_rate: Learning rate for optimizer
        public_loader: Optional DataLoader for public pseudo-labeled data

    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss()

    # Select optimizer
    if optimizer_name.upper() == "SGD":
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
    elif optimizer_name.upper() == "ADAM":
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=learning_rate,
            weight_decay=5e-4
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported: SGD, Adam")

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for epoch in range(epochs):
        # If we have public data, create iterator for it
        public_iter = iter(public_loader) if public_loader is not None else None

        for data, target in trainloader:
            data, target = data.to(device), target.to(device)

            # If public data available, sample a batch from it too
            if public_iter is not None:
                try:
                    public_data, public_target = next(public_iter)
                    public_data, public_target = public_data.to(device), public_target.to(device)

                    # Combine private and public batches
                    data = torch.cat([data, public_data], dim=0)
                    target = torch.cat([target, public_target], dim=0)
                except StopIteration:
                    # Public loader exhausted, restart it
                    public_iter = iter(public_loader)
                    try:
                        public_data, public_target = next(public_iter)
                        public_data, public_target = public_data.to(device), public_target.to(device)
                        data = torch.cat([data, public_data], dim=0)
                        target = torch.cat([target, public_target], dim=0)
                    except:
                        pass  # Continue with just private data

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def test(net, testloader, device: torch.device):
    """
    Test the network

    Args:
        net: Neural network model
        testloader: DataLoader with test data
        device: Device to test on (cuda/cpu)

    Returns:
        avg_loss: Average test loss
        accuracy: Test accuracy
    """
    net.to(device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(testloader)
    accuracy = correct / total

    return avg_loss, accuracy


def predict_on_unlabeled(net, public_dataset, device: torch.device) -> List[int]:
    """
    Make hard label predictions on public unlabeled dataset

    Args:
        net: Neural network model
        public_dataset: Unlabeled public dataset U
        device: Device to use

    Returns:
        List of predicted class labels (hard labels)
    """
    net.to(device)
    net.eval()

    dataloader = DataLoader(
        public_dataset,
        batch_size=len(public_dataset),
        shuffle=False,
        num_workers=0
    )

    predictions = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = net(data)
            _, predicted = output.max(1)
            predictions.extend(predicted.cpu().numpy().tolist())

    return predictions


# ============================================================================
# PSEUDO-LABELING
# ============================================================================

def create_pseudo_labeled_dataset(public_dataset, consensus_labels: List[int]):
    """
    Create pseudo-labeled dataset from public dataset and consensus labels

    Args:
        public_dataset: Unlabeled public dataset
        consensus_labels: Consensus labels from server

    Returns:
        PseudoLabeledDataset with consensus labels
    """
    images = [public_dataset[i] for i in range(len(public_dataset))]
    return PseudoLabeledDataset(images, consensus_labels)


def combine_with_pseudo_labels(trainloader, pseudo_dataset, public_batch_size=32):
    """
    Create a combined training approach with separate batch sampling.

    This returns two dataloaders:
    1. Private data loader (for sampling private batches)
    2. Public pseudo-labeled data loader (for sampling public batches)

    Args:
        trainloader: DataLoader with private training data
        pseudo_dataset: PseudoLabeledDataset with consensus labels
        public_batch_size: Batch size for public data sampling

    Returns:
        Tuple of (private_loader, public_loader) for mixed batch training
    """
    # Private data loader (already exists)
    private_loader = trainloader

    # Public pseudo-labeled data loader
    public_loader = DataLoader(
        pseudo_dataset,
        batch_size=public_batch_size,
        shuffle=True,
        num_workers=0
    )

    return private_loader, public_loader