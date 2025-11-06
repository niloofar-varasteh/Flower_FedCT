import argparse, random, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# -----------------------
# Utils
# -----------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(num_classes):
    # یک CNN ساده و سبک که روی هر دو دیتاست جواب می‌دهد
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
        nn.Linear(256, num_classes)
    )

def build_model_gray(num_classes):
    # برای FashionMNIST (1 کاناله)
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Linear(128, num_classes)
    )

def get_datasets(name):
    if name == "CIFAR10":
        tf_train = T.Compose([T.ToTensor()])
        tf_test  = T.Compose([T.ToTensor()])
        train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tf_train)
        test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tf_test)
        num_classes, channels = 10, 3
    elif name == "FashionMNIST":
        tf_train = T.Compose([T.ToTensor()])
        tf_test  = T.Compose([T.ToTensor()])
        train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf_train)
        test  = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=tf_test)
        num_classes, channels = 10, 1
    else:
        raise ValueError("Dataset must be CIFAR10 or FashionMNIST")
    return train, test, num_classes, channels

def iid_partition(n_samples, num_clients):
    # تقسیم یکنواخت اندیس‌ها بین کلاینت‌ها
    idxs = np.random.permutation(n_samples)
    splits = np.array_split(idxs, num_clients)
    return [list(s) for s in splits]

def evaluate(model, loader, device_):
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
    return loss_sum/total, correct/total

def train_one_epoch(model, loader, optimizer, device_):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device_), y.to(device_)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

def get_optimizer(name, params, lr):
    if name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif name.lower() == "adam":
        return torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError("Optimizer must be SGD or Adam")

def get_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def set_state(model, state):
    model.load_state_dict(state, strict=True)

def average_states(states):
    # FedAvg وزن‌ها را میانگین می‌گیرد
    avg = {}
    for k in states[0].keys():
        avg[k] = sum(s[k] for s in states) / len(states)
    return avg

# -----------------------
# Main FedAvg
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication-rounds", type=int, default=15)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--local-rounds", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    # آرگومان‌های بدون استفاده، فقط برای هم‌فرمی با FedCT:
    parser.add_argument("--unlabeled", type=int, default=100)
    parser.add_argument("--private-batch-size", type=int, default=64)
    parser.add_argument("--public-batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed(42)
    dev = device()

    # Load data
    train_ds, test_ds, num_classes, channels = get_datasets(args.dataset)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # Partition train indices IID
    parts = iid_partition(len(train_ds), args.clients)

    # Prepare client loaders once (ثابت و قابل استفاده در همه‌ی راندها)
    client_loaders = []
    for i in range(args.clients):
        subset = Subset(train_ds, parts[i])
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    # Init global model
    if channels == 3:
        global_model = build_model(num_classes).to(dev)
    else:
        global_model = build_model_gray(num_classes).to(dev)

    # Evaluate initial
    init_loss, init_acc = evaluate(global_model, test_loader, dev)
    print(f"Round 0 - Loss: {init_loss:.4f} - Accuracy: {init_acc:.4f}")

    # FedAvg loop
    for rnd in range(1, args.communication_rounds + 1):
        client_states = []

        for cid in range(args.clients):
            # Clone global -> local
            local_model = build_model(num_classes).to(dev) if channels==3 else build_model_gray(num_classes).to(dev)
            set_state(local_model, get_state(global_model))

            optimizer = get_optimizer(args.optimizer, local_model.parameters(), args.lr)
            # local training: args.local_rounds epoch روی داده‌ی کلاینت
            for _ in range(args.local_rounds):
                train_one_epoch(local_model, client_loaders[cid], optimizer, dev)

            # collect weights
            client_states.append(get_state(local_model))

        # average
        new_global = average_states(client_states)
        set_state(global_model, new_global)

        # evaluate
        loss, acc = evaluate(global_model, test_loader, dev)
        print(f"Round {rnd} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")

    # summary lines (برای grep در اسکریپت bash)
    print(f"Average Test Loss: {loss:.4f}")
    print(f"Average Test Acc: {acc:.4f}")

if __name__ == "__main__":
    main()
