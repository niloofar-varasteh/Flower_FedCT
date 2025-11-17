#!/usr/bin/env python3
"""
Federated Averaging (FedAvg) implementation.
This matches FedCT parameters for fair comparison.
Now supports: resnet18, resnet34, cnn_small, LightCNN.
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from torchvision import models as tvm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cnn_small(num_classes, in_ch=3):
    """Simple CNN baseline."""
    if in_ch == 3:
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )


def build_lightcnn(num_classes, in_ch=3):
    """
    A lightweight CNN: deeper than cnn_small, but much smaller than ResNet18.
    Works for both RGB (in_ch=3) and grayscale (in_ch=1).
    """
    return nn.Sequential(
        # Block 1
        nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # 32x32 -> 16x16 (CIFAR-style)

        # Block 2
        nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # 16x16 -> 8x8

        # Block 3
        nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),  # -> 1x1

        nn.Flatten(),
        nn.Linear(128, num_classes),
    )


def _make_resnet_cifar(backbone: str, num_classes: int, in_ch: int):
    """CIFAR-style ResNet."""
    if backbone == "resnet18":
        m = tvm.resnet18(weights=None)
    elif backbone == "resnet34":
        m = tvm.resnet34(weights=None)
    else:
        raise ValueError("Unsupported ResNet backbone")

    # Replace first conv for CIFAR size and in_ch
    m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    # Classifier
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_model(arch: str, num_classes: int, in_ch: int):
    """Build model based on architecture name."""
    arch = arch.lower()  # so LightCNN/LIGHTCNN/etc. all map to 'lightcnn'
    if arch == "cnn_small":
        return build_cnn_small(num_classes, in_ch)
    elif arch in ("resnet18", "resnet34"):
        return _make_resnet_cifar(arch, num_classes, in_ch)
    elif arch == "lightcnn":
        return build_lightcnn(num_classes, in_ch)
    else:
        raise ValueError("Unknown --arch (use: resnet18, resnet34, cnn_small, LightCNN)")


def get_datasets(name):
    """Load dataset with minimal transforms."""
    if name == "CIFAR10":
        tf = T.Compose([T.ToTensor()])
        train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
        test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)
        num_classes, in_ch = 10, 3
    elif name == "FashionMNIST":
        tf = T.Compose([T.ToTensor()])
        train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf)
        test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=tf)
        num_classes, in_ch = 10, 1
    else:
        raise ValueError("Dataset must be CIFAR10 or FashionMNIST")
    return train, test, num_classes, in_ch


def iid_partition(n_samples, num_clients):
    """Partition dataset into IID splits."""
    idxs = np.random.permutation(n_samples)
    splits = np.array_split(idxs, num_clients)
    return [list(s) for s in splits]


def iid_partition_with_public(n_samples, num_clients, public_size=100, seed=42):
    """
    Partition dataset matching FedCT's split:
    - First `public_size` samples are separated as public dataset (shared by all clients)
    - Remaining samples are split IID among clients
    
    This ensures fair comparison with FedCT which uses:
    - 100 public samples (with pseudo-labels)
    - Remaining samples split among clients (with true labels)
    
    FedAvg will use:
    - Same 100 public samples (with TRUE labels)
    - Same remaining samples split among clients
    """
    np.random.seed(seed)
    all_indices = np.arange(n_samples)
    
    # Sample indices for public dataset (same as FedCT does)
    public_indices = np.random.choice(all_indices, size=public_size, replace=False)
    
    # Remaining indices for private data
    private_mask = np.ones(n_samples, dtype=bool)
    private_mask[public_indices] = False
    private_indices = all_indices[private_mask]
    
    # Partition private data among clients (IID split)
    samples_per_client = len(private_indices) // num_clients
    client_splits = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_private = private_indices[start_idx:end_idx]
        # Each client gets their private data + all public data
        client_splits.append(np.concatenate([client_private, public_indices]).tolist())
    
    return client_splits, public_indices.tolist()


def evaluate(model, loader, device_):
    """Evaluate model on dataset."""
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
    """Train model for one epoch. Returns average train loss and accuracy."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device_), y.to(device_)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        # Track training metrics
        loss_sum += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)
    
    return loss_sum / total, correct / total


def get_optimizer(name, params, lr):
    """Get optimizer by name."""
    if name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif name.lower() == "adam":
        return torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError("Optimizer must be SGD or Adam")


def get_state(model):
    """Get model state dict."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_state(model, state):
    """Set model state dict."""
    model.load_state_dict(state, strict=True)


def average_states(states):
    """Average multiple state dicts (FedAvg)."""
    avg = {}
    for k in states[0].keys():
        # Only average floating point tensors
        if states[0][k].dtype in [torch.float32, torch.float64]:
            avg[k] = torch.stack([s[k].float() for s in states]).mean(dim=0).type_as(states[0][k])
        else:
            # For non-float tensors, use the first one
            avg[k] = states[0][k].clone()
    return avg


def main():
    parser = argparse.ArgumentParser(description="Federated Averaging (FedAvg)")
    parser.add_argument("--communication-rounds", type=int, default=15, help="Aggregation rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--local-rounds", type=int, default=3, help="Local epochs per aggregation")
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "FashionMNIST"])
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "cnn_small", "LightCNN"],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--log-per-local",
        type=int,
        default=1,
        help="If 1, log avg client accuracy after each local epoch",
    )
    parser.add_argument(
        "--public-size",
        type=int,
        default=0,
        help="If >0, use fair split with public dataset (matching FedCT). Default 0 = use all data.",
    )

    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)
    dev = device()

    print("=" * 80)
    print("FEDERATED AVERAGING (FedAvg)")
    print("=" * 80)
    print(f"Dataset:              {args.dataset}")
    print(f"Clients:              {args.clients}")
    print(f"Aggregation rounds:   {args.communication_rounds}")
    print(f"Local epochs/round:   {args.local_rounds}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Optimizer:            {args.optimizer}")
    print(f"Learning rate:        {args.lr}")
    print(f"Architecture:         {args.arch}")
    print(f"Seed:                 {args.seed}")
    print(f"Log per local round:  {args.log_per_local}")
    if args.public_size > 0:
        print(f"Public dataset size:  {args.public_size} (FAIR COMPARISON MODE)")
    print(f"Using device:         {dev}")
    print("=" * 80)
    print()

    # Data
    train_ds, test_ds, num_classes, in_ch = get_datasets(args.dataset)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # IID split - fair comparison mode or standard mode
    if args.public_size > 0:
        # Fair comparison: match FedCT's data split
        parts, public_indices = iid_partition_with_public(
            len(train_ds), args.clients, args.public_size, args.seed
        )
        print(f"FAIR COMPARISON MODE:")
        print(f"  - Public dataset: {args.public_size} samples (shared, with TRUE labels)")
        print(f"  - Private data per client: ~{len(parts[0]) - args.public_size} samples")
        print(f"  - Total per client: ~{len(parts[0])} samples\n")
    else:
        # Standard mode: split all data among clients
        parts = iid_partition(len(train_ds), args.clients)
        print(f"STANDARD MODE:")
        print(f"  - Data partitioned: {args.clients} clients, ~{len(parts[0])} samples each\n")
    
    client_loaders = [
        DataLoader(Subset(train_ds, part), batch_size=args.batch_size, shuffle=True, num_workers=0)
        for part in parts
    ]

    # Init global model
    global_model = build_model(args.arch, num_classes, in_ch).to(dev)

    # Round 1 eval
    init_loss, init_acc = evaluate(global_model, test_loader, dev)
    print(f"Round 1 - Loss: {init_loss:.4f} - Accuracy: {init_acc:.4f}")

    # FedAvg training
    for rnd in range(1, args.communication_rounds + 1):
        print(f"\n{'=' * 80}")
        print(f"Aggregation Round {rnd}/{args.communication_rounds}")
        print(f"{'=' * 80}")

        # Create per-client models cloned from global (once per round)
        local_models = [build_model(args.arch, num_classes, in_ch).to(dev) for _ in range(args.clients)]
        for m in local_models:
            set_state(m, get_state(global_model))
        opts = [get_optimizer(args.optimizer, m.parameters(), args.lr) for m in local_models]

        # Train for local epochs, and (optionally) log after EACH local epoch
        for e in range(1, args.local_rounds + 1):
            # 1) One local epoch on each client, collect train metrics
            train_losses, train_accs = [], []
            for cid, (m, opt, loader) in enumerate(zip(local_models, opts, client_loaders)):
                train_loss, train_acc = train_one_epoch(m, loader, opt, dev)
                train_losses.append(train_loss)
                train_accs.append(train_acc)

            if args.log_per_local:
                # 2) After finishing this local epoch on ALL clients,
                #    evaluate each client's model on the common test set
                test_accs, test_losses = [], []
                for m in local_models:
                    l, a = evaluate(m, test_loader, dev)
                    test_losses.append(l)
                    test_accs.append(a)

                # 3) Average across clients -> a single point per LOCAL epoch
                avg_train_acc = float(np.mean(train_accs))
                avg_train_loss = float(np.mean(train_losses))
                avg_test_acc = float(np.mean(test_accs))
                avg_test_loss = float(np.mean(test_losses))

                # 4) IMPORTANT: Keep this exact format so compare_fedct_fedavg.py can parse it
                #    Format: [LOCAL] epoch=X/Y train_acc=... train_loss=... test_acc=... test_loss=...
                print(f"[LOCAL] epoch={e}/{args.local_rounds} train_acc={avg_train_acc:.4f} train_loss={avg_train_loss:.4f} test_acc={avg_test_acc:.4f} test_loss={avg_test_loss:.4f}")

        # Aggregate client models
        print("\n  Aggregating client models...")
        new_global = average_states([get_state(m) for m in local_models])
        set_state(global_model, new_global)
        print("  [OK] Global model updated")

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
