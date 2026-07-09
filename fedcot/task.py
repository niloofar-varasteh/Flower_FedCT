"""
ML Task: Model, Data, Training for FedCT , task.py = model.py + dataset.py + Training Functions
Contains all ML-related code: model definition, data loading, training, evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import numpy as np
import random
from typing import List, Tuple


# ============================================================================
# REPRODUCIBILITY UTILITIES
# ============================================================================

def set_global_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set global seed for data loading and other operations
# Client-specific seeds will be set in client_app.py for model initialization
set_global_seed(42)


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class BasicBlock(nn.Module):
    """Basic Residual Block for ResNet"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ImprovedCNN(nn.Module):
    """Improved ResNet-style CNN for CIFAR-10/FashionMNIST with better accuracy"""

    def __init__(self, num_classes=10, in_channels=3):
        super(ImprovedCNN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class LightCNN(nn.Module):
    """Lightweight CNN optimized for FashionMNIST/CIFAR-10 with ~400K parameters

    Architecture:
        - 3 convolutional blocks with batch normalization
        - Max pooling for downsampling
        - Adaptive average pooling for flexible input sizes
        - Dropout for regularization
        - Much faster training than ResNet-18 (~30x fewer parameters)
        - Expected accuracy: 88-92% on FashionMNIST, 75-80% on CIFAR-10
    """

    def __init__(self, num_classes=10, in_channels=1):
        super(LightCNN, self).__init__()

        # First conv block: in_channels -> 32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14 (or 32x32 -> 16x16)

        # Second conv block: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7 (or 16x16 -> 8x8)

        # Third conv block: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 7x7 -> 3x3 (or 8x8 -> 4x4)

        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # Ensures 4x4 output regardless of input

        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # 128 channels * 4x4 spatial
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Conv block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Conv block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Adaptive pooling
        x = self.avgpool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def get_model(dataset: str = "CIFAR10", architecture: str = "ResNet18"):
    """Get model instance based on dataset and architecture

    Args:
        dataset: Dataset name - "CIFAR10" or "FashionMNIST"
        architecture: Model architecture - "ResNet18" or "LightCNN"

    Returns:
        Model instance
    """
    # Determine input channels
    in_channels = 3 if dataset == "CIFAR10" else 1
    num_classes = 10

    # Select architecture
    if architecture.upper() == "RESNET18":
        return ImprovedCNN(num_classes=num_classes, in_channels=in_channels)
    elif architecture.upper() == "LIGHTCNN":
        return LightCNN(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Use 'ResNet18' or 'LightCNN'")


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
# DATA LOADING
# ============================================================================

def load_data(partition_id: int, num_partitions: int = 5, unlabeled_size: int = 500, batch_size: int = 32,
              dataset: str = "CIFAR10"):
    """
    Load dataset for a specific partition

    Args:
        partition_id: ID of this client (0 to num_partitions-1)
        num_partitions: Total number of clients
        unlabeled_size: Size of public unlabeled dataset U (shared across all clients)
        batch_size: Batch size for training
        dataset: Dataset name - "CIFAR10" or "FashionMNIST"
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
        public_base = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_test
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
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
        public_base = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform_test
        )
        testset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform_test
        )
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

    # Create a deterministic public unlabeled dataset U. Public predictions must
    # refer to the same samples/views across clients and communication rounds.
    public_dataset = UnlabeledDataset(public_base, public_indices)

    # Partition private data among clients (IID split)
    samples_per_client = len(private_indices) // num_partitions
    start_idx = partition_id * samples_per_client
    end_idx = start_idx + samples_per_client
    client_indices = private_indices[start_idx:end_idx]

    # Create dataloaders
    # Note: num_workers=0 for Ray compatibility in simulation mode
    client_trainset = Subset(trainset, client_indices)
    trainloader = DataLoader(client_trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    # Print detailed data statistics (suppressed to avoid Ray actor clutter)
    # Only printed in direct execution mode, not in Flower simulation
    # print(f"✓ Client {partition_id + 1} Data Summary:")
    # print(f"  - Dataset: {dataset}")
    # print(f"  - Private training samples: {len(client_trainset)}")
    # print(f"  - Public unlabeled samples (shared): {unlabeled_size}")
    # print(f"  - Test samples (full test set): {len(testset)}")
    # print(f"  - Total training samples: {len(trainset)}")
    # print(f"  - Samples used for public unlabeled: {unlabeled_size}")
    # print(f"  - Samples distributed to {num_partitions} clients: {len(private_indices)}")

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

    # Note: num_workers=0 for Ray compatibility in simulation mode
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
    Create a single loader over D_i union P.

    Args:
        trainloader: DataLoader with private training data
        pseudo_dataset: PseudoLabeledDataset with consensus labels
        public_batch_size: Kept for backward-compatible call sites.

    Returns:
        DataLoader over the union of private and pseudo-labeled public data.
    """
    combined_dataset = ConcatDataset([trainloader.dataset, pseudo_dataset])
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=trainloader.batch_size,
        shuffle=True,
        num_workers=0
    )

    return combined_loader
