"""Single-file Flower FedAvg baseline.

This app intentionally keeps FedAvg separate from FedCT. Flower provides the
server-side FedAvg strategy; this file only provides the baseline client model,
data pipeline, and training/evaluation loop needed to compare against FedCT.
"""

from pathlib import Path
import logging
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, FitIns, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader, Subset
from torchvision import models as tvm


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lightcnn(num_classes, in_ch=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, num_classes),
    )


def build_cnn_small(num_classes, in_ch=3):
    if in_ch == 3:
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Linear(128, num_classes),
    )


def make_resnet_cifar(arch, num_classes, in_ch):
    if arch == "resnet18":
        model = tvm.resnet18(weights=None)
    elif arch == "resnet34":
        model = tvm.resnet34(weights=None)
    else:
        raise ValueError(f"Unsupported ResNet architecture: {arch}")
    model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(architecture, num_classes, in_ch):
    arch = architecture.lower()
    if arch == "lightcnn":
        return build_lightcnn(num_classes, in_ch)
    if arch == "cnn_small":
        return build_cnn_small(num_classes, in_ch)
    if arch in ("resnet18", "resnet34"):
        return make_resnet_cifar(arch, num_classes, in_ch)
    raise ValueError("architecture must be LightCNN, ResNet18, ResNet34, or cnn_small")


def get_datasets(name):
    transform = T.Compose([T.ToTensor()])
    if name == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(
            root=str(DATA_DIR), train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=str(DATA_DIR), train=False, download=True, transform=transform
        )
        return trainset, testset, 10, 3
    if name == "FashionMNIST":
        trainset = torchvision.datasets.FashionMNIST(
            root=str(DATA_DIR), train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.FashionMNIST(
            root=str(DATA_DIR), train=False, download=True, transform=transform
        )
        return trainset, testset, 10, 1
    raise ValueError("dataset must be CIFAR10 or FashionMNIST")


def partition_indices(n_samples, num_clients, unlabeled_size, seed):
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_samples)
    public_indices = rng.choice(all_indices, size=unlabeled_size, replace=False)
    private_mask = np.ones(n_samples, dtype=bool)
    private_mask[public_indices] = False
    private_indices = all_indices[private_mask]

    samples_per_client = len(private_indices) // num_clients
    parts = []
    for cid in range(num_clients):
        start = cid * samples_per_client
        end = start + samples_per_client
        parts.append(private_indices[start:end].tolist())
    return parts


def get_optimizer(name, parameters, learning_rate):
    if name.upper() == "SGD":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    if name.upper() == "ADAM":
        return torch.optim.Adam(parameters, lr=learning_rate)
    raise ValueError("optimizer must be SGD or Adam")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * y.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += y.size(0)
    return loss_sum / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += y.size(0)
    return loss_sum / total, correct / total


def get_parameters(model):
    return [value.detach().cpu().numpy() for _, value in model.state_dict().items()]


def set_parameters(model, parameters):
    state_dict = model.state_dict()
    params_dict = {
        key: torch.tensor(value)
        for key, value in zip(state_dict.keys(), parameters)
    }
    model.load_state_dict(params_dict, strict=True)


class FedAvgBaselineClient(NumPyClient):
    def __init__(self, context):
        self.num_clients = int(context.run_config.get("num-clients", 5))
        self.partition_id = int(
            context.node_config.get(
                "partition-id",
                context.node_config.get("partition_id", context.node_id % self.num_clients),
            )
        )
        self.dataset = context.run_config.get("dataset", "CIFAR10")
        self.architecture = context.run_config.get("architecture", "LightCNN")
        self.batch_size = int(context.run_config.get("batch-size", 64))
        self.unlabeled_size = int(context.run_config.get("unlabeled-size", 0))
        self.seed = int(context.run_config.get("seed", 42))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(self.seed + self.partition_id)
        trainset, testset, num_classes, in_ch = get_datasets(self.dataset)
        parts = partition_indices(
            len(trainset), self.num_clients, self.unlabeled_size, self.seed
        )
        self.trainloader = DataLoader(
            Subset(trainset, parts[self.partition_id]),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
        self.model = build_model(self.architecture, num_classes, in_ch).to(self.device)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        if parameters:
            set_parameters(self.model, parameters)

        optimizer_name = str(config.get("optimizer", "Adam"))
        learning_rate = float(config.get("learning_rate", 0.001))
        local_epochs = int(config.get("num_local_rounds", 1))
        optimizer = get_optimizer(optimizer_name, self.model.parameters(), learning_rate)

        metrics = {}
        last_train_loss, last_train_acc = 0.0, 0.0
        last_test_loss, last_test_acc = 0.0, 0.0
        for epoch in range(1, local_epochs + 1):
            last_train_loss, last_train_acc = train_one_epoch(
                self.model, self.trainloader, optimizer, self.device
            )
            last_test_loss, last_test_acc = evaluate(
                self.model, self.testloader, self.device
            )
            metrics[f"train_loss_epoch_{epoch}"] = float(last_train_loss)
            metrics[f"train_acc_epoch_{epoch}"] = float(last_train_acc)
            metrics[f"test_loss_epoch_{epoch}"] = float(last_test_loss)
            metrics[f"test_acc_epoch_{epoch}"] = float(last_test_acc)

        metrics.update(
            {
                "train_loss": float(last_train_loss),
                "train_acc": float(last_train_acc),
                "test_loss": float(last_test_loss),
                "test_acc": float(last_test_acc),
            }
        )
        return get_parameters(self.model), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        if parameters:
            set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"test_acc": float(acc)}


def client_fn(context):
    return FedAvgBaselineClient(context).to_client()


class LoggingFedAvg(FedAvg):
    def __init__(
        self,
        num_clients,
        num_communication_rounds,
        num_local_rounds,
        optimizer,
        learning_rate,
        initial_parameters,
    ):
        self.num_clients = int(num_clients)
        self.num_communication_rounds = int(num_communication_rounds)
        self.num_local_rounds = int(num_local_rounds)
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=self.num_clients,
            min_evaluate_clients=self.num_clients,
            min_available_clients=self.num_clients,
            initial_parameters=initial_parameters,
            accept_failures=False,
        )

    def configure_fit(self, server_round, parameters, client_manager):
        configured = super().configure_fit(server_round, parameters, client_manager)
        fit_config = []
        for client, fit_ins in configured:
            config = dict(fit_ins.config)
            config.update(
                {
                    "optimizer": self.optimizer,
                    "learning_rate": self.learning_rate,
                    "num_local_rounds": self.num_local_rounds,
                }
            )
            fit_config.append((client, FitIns(parameters=fit_ins.parameters, config=config)))
        return fit_config

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if results:
            logger.info(f"\nAggregation Round {server_round}/{self.num_communication_rounds}")
            for epoch in range(1, self.num_local_rounds + 1):
                train_losses, train_accs, test_losses, test_accs = [], [], [], []
                for _, fit_res in results:
                    m = fit_res.metrics
                    train_losses.append(float(m.get(f"train_loss_epoch_{epoch}", 0.0)))
                    train_accs.append(float(m.get(f"train_acc_epoch_{epoch}", 0.0)))
                    test_losses.append(float(m.get(f"test_loss_epoch_{epoch}", 0.0)))
                    test_accs.append(float(m.get(f"test_acc_epoch_{epoch}", 0.0)))
                logger.info(
                    f"[LOCAL] epoch={epoch}/{self.num_local_rounds} "
                    f"train_acc={np.mean(train_accs):.4f} "
                    f"train_loss={np.mean(train_losses):.4f} "
                    f"test_acc={np.mean(test_accs):.4f} "
                    f"test_loss={np.mean(test_losses):.4f}"
                )

            final_accs = [float(fit_res.metrics.get("test_acc", 0.0)) for _, fit_res in results]
            final_losses = [
                float(fit_res.metrics.get("test_loss", 0.0)) for _, fit_res in results
            ]
            logger.info(
                f"Round {server_round} - Loss: {np.mean(final_losses):.4f} "
                f"- Accuracy: {np.mean(final_accs):.4f}"
            )
        return parameters, metrics


def server_fn(context: Context):
    num_communication_rounds = int(context.run_config.get("num-communication-rounds", 10))
    num_clients = int(context.run_config.get("num-clients", 5))
    num_local_rounds = int(context.run_config.get("num-local-rounds", 2))
    dataset = context.run_config.get("dataset", "CIFAR10")
    architecture = context.run_config.get("architecture", "LightCNN")
    optimizer = context.run_config.get("optimizer", "Adam")
    learning_rate = float(context.run_config.get("learning-rate", 0.001))
    unlabeled_size = int(context.run_config.get("unlabeled-size", 0))
    seed = int(context.run_config.get("seed", 42))

    logger.info(f"\n{'=' * 80}")
    logger.info("FEDERATED AVERAGING (FedAvg) - FLOWER BUILT-IN STRATEGY ONLY")
    logger.info(f"{'=' * 80}")
    logger.info("Configuration:")
    logger.info(f"  Dataset:                       {dataset}")
    logger.info(f"  Architecture:                  {architecture}")
    logger.info(f"  Aggregation rounds:            {num_communication_rounds}")
    logger.info(f"  Number of clients:             {num_clients}")
    logger.info(f"  Local epochs per aggregation:  {num_local_rounds}")
    logger.info(f"  Optimizer:                     {optimizer}")
    logger.info(f"  Learning rate:                 {learning_rate}")
    logger.info(f"  Unlabeled samples excluded:    {unlabeled_size}")
    logger.info(f"  Seed:                          {seed}")
    logger.info("  Baseline pipeline:             weak ToTensor-only, old run_fedavg style")
    logger.info(f"{'=' * 80}\n")

    set_seed(seed)
    _, _, num_classes, in_ch = get_datasets(dataset)
    initial_model = build_model(architecture, num_classes, in_ch)
    strategy = LoggingFedAvg(
        num_clients=num_clients,
        num_communication_rounds=num_communication_rounds,
        num_local_rounds=num_local_rounds,
        optimizer=optimizer,
        learning_rate=learning_rate,
        initial_parameters=ndarrays_to_parameters(get_parameters(initial_model)),
    )
    config = ServerConfig(num_rounds=num_communication_rounds, round_timeout=None)
    return ServerAppComponents(strategy=strategy, config=config)


client_app = ClientApp(client_fn=client_fn)
server_app = ServerApp(server_fn=server_fn)
