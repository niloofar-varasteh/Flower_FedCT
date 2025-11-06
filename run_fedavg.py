#!/usr/bin/env python3
"""
Federated Averaging (FedAvg) Implementation
Standard federated learning with model aggregation but NO pseudo-labeling.
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


# -----------------------
# Utils
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes):
    """Simple CNN for CIFAR10"""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
        nn.Linear(256, num_classes)
    )


def build_model_gray(num_classes):
    """Simple CNN for FashionMNIST (1 channel)"""
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Linear(128, num_classes)
    )


def get_datasets(name):
    """Load dataset"""
    if name == "CIFAR10":
        tf = T.Compose([T.ToTensor()])
        train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
        test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)
        num_classes, channels = 10, 3
    elif name == "FashionMNIST":
        tf = T.Compose([T.ToTensor()])
        train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf)
        test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=tf)
        num_classes, channels = 10, 1
    else:
        raise ValueError("Dataset must be CIFAR10 or FashionMNIST")
    return train, test, num_classes, channels


def iid_partition(n_samples, num_clients):
    """Split dataset indices uniformly among clients"""
    idxs = np.random.permutation(n_samples)
    splits = np.array_split(idxs, num_clients)
    return [list(s) for s in splits]


def evaluate(model, loader, device_):
    """Evaluate model on test set"""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device_), y.to(device_)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, optimizer, device_):
    """Train model for one epoch"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device_), y.to(device_)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


def get_optimizer(name, params, lr):
    """Get optimizer by name"""
    if name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif name.lower() == "adam":
        return torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError("Optimizer must be SGD or Adam")


def get_state(model):
    """Get model state dict"""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_state(model, state):
    """Set model state dict"""
    model.load_state_dict(state, strict=True)


def average_states(states):
    """Average model weights (FedAvg aggregation)"""
    avg = {}
    for k in states[0].keys():
        avg[k] = sum(s[k] for s in states) / len(states)
    return avg


# -----------------------
# Main FedAvg
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Federated Averaging (FedAvg)")
    parser.add_argument("--communication-rounds", type=int, default=15,
                        help="Number of communication rounds")
    parser.add_argument("--clients", type=int, default=5,
                        help="Number of clients")
    parser.add_argument("--local-rounds", type=int, default=3,
                        help="Number of local epochs per communication round")
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        choices=["CIFAR10", "FashionMNIST"],
                        help="Dataset to use")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        choices=["SGD", "Adam"],
                        help="Optimizer to use")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")

    # Unused arguments for compatibility with FedCT scripts
    parser.add_argument("--unlabeled", type=int, default=100)
    parser.add_argument("--private-batch-size", type=int, default=64)
    parser.add_argument("--public-batch-size", type=int, default=32)

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(42)
    dev = device()

    print("=" * 80)
    print("FEDERATED AVERAGING (FedAvg)")
    print("=" * 80)
    print(f"Dataset:              {args.dataset}")
    print(f"Clients:              {args.clients}")
    print(f"Communication rounds: {args.communication_rounds}")
    print(f"Local epochs/round:   {args.local_rounds}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Optimizer:            {args.optimizer}")
    print(f"Learning rate:        {args.lr}")
    print("=" * 80)
    print()

    # Load data
    train_ds, test_ds, num_classes, channels = get_datasets(args.dataset)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # Partition training data IID among clients
    parts = iid_partition(len(train_ds), args.clients)

    # Create client data loaders
    client_loaders = []
    for i in range(args.clients):
        subset = Subset(train_ds, parts[i])
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    print(f"Data partitioned: {args.clients} clients, ~{len(parts[0])} samples each")
    print()

    # Initialize global model
    if channels == 3:
        global_model = build_model(num_classes).to(dev)
    else:
        global_model = build_model_gray(num_classes).to(dev)

    # Evaluate initial model
    init_loss, init_acc = evaluate(global_model, test_loader, dev)
    print(f"Round 0 - Loss: {init_loss:.4f} - Accuracy: {init_acc:.4f}")

    # FedAvg training loop
    for rnd in range(1, args.communication_rounds + 1):
        print(f"\n{'=' * 80}")
        print(f"Communication Round {rnd}/{args.communication_rounds}")
        print(f"{'=' * 80}")

        client_states = []

        # Train each client
        for cid in range(args.clients):
            # Create local model as copy of global
            if channels == 3:
                local_model = build_model(num_classes).to(dev)
            else:
                local_model = build_model_gray(num_classes).to(dev)

            set_state(local_model, get_state(global_model))

            # Create optimizer for local training
            optimizer = get_optimizer(args.optimizer, local_model.parameters(), args.lr)

            # Local training: multiple epochs on client's data
            for epoch in range(args.local_rounds):
                train_one_epoch(local_model, client_loaders[cid], optimizer, dev)

            # Collect local model weights
            client_states.append(get_state(local_model))

            print(f"  Client {cid + 1}/{args.clients} completed local training")

        # Aggregate models (FedAvg)
        print("\n  Aggregating client models...")
        new_global = average_states(client_states)
        set_state(global_model, new_global)
        print("  ✓ Global model updated")

        # Evaluate global model
        loss, acc = evaluate(global_model, test_loader, dev)
        print(f"\nRound {rnd} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Average Test Loss: {loss:.4f}")
    print(f"Average Test Acc:  {acc:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()