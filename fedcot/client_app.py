"""
Flower Client App for FedCT.

Clients keep local models, train on private data plus consensus pseudo-labels,
and share only hard labels on the public unlabeled dataset.
"""

import logging

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import ArrayRecord, Context, RecordDict

from fedcot.task import (
    combine_with_pseudo_labels,
    create_pseudo_labeled_dataset,
    get_model,
    load_data,
    predict_on_unlabeled,
    test,
    train,
)

logger = logging.getLogger(__name__)


def normalize_architecture(arch: str) -> str:
    """Normalize architecture name."""
    if arch.lower() == "lightcnn":
        return "LightCNN"
    return "ResNet18"


class FedCTClient(NumPyClient):
    """Flower client for Federated Co-Training."""

    def __init__(
        self,
        partition_id: int,
        num_partitions: int,
        unlabeled_size: int = 500,
        dataset: str = "CIFAR10",
        architecture: str = "ResNet18",
        private_batch_size: int = 32,
        public_batch_size: int = 32,
        state: RecordDict = None,
    ):
        self.partition_id = int(partition_id)
        self.num_partitions = int(num_partitions)
        self.unlabeled_size = int(unlabeled_size)
        self.dataset = dataset
        self.architecture = normalize_architecture(architecture)
        self.private_batch_size = int(private_batch_size)
        self.public_batch_size = int(public_batch_size)
        self.state = state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trainloader = None
        self.valloader = None
        self.public_dataset = None
        self.pseudo_dataset = None
        self.net = None
        self.loaded_partition_id = None

    @property
    def state_key(self) -> str:
        """Key used to persist this client's local model."""
        return (
            f"fedct_model_{self.dataset}_{self.architecture}"
            f"_partition_{self.partition_id}"
        )

    def _ensure_loaded(self):
        if self.net is not None and self.loaded_partition_id == self.partition_id:
            return

        self.trainloader, self.valloader, self.public_dataset = load_data(
            partition_id=self.partition_id,
            num_partitions=self.num_partitions,
            unlabeled_size=self.unlabeled_size,
            dataset=self.dataset,
            batch_size=self.private_batch_size,
        )
        self.net = get_model(dataset=self.dataset, architecture=self.architecture)
        self.loaded_partition_id = self.partition_id

        if self.state is not None:
            saved_state = self.state.get(self.state_key)
            if saved_state is not None:
                self.net.load_state_dict(saved_state.to_torch_state_dict())

    def _save_state(self):
        """Persist local model weights across ClientApp recreations."""
        if self.state is not None:
            self.state[self.state_key] = ArrayRecord(self.net.state_dict())

    def fit(self, parameters, config):
        try:
            self._ensure_loaded()
            trainloader = self.trainloader

            if "consensus_labels" in config:
                consensus_labels = [
                    int(x) for x in config["consensus_labels"].split(",") if x != ""
                ]
                self.pseudo_dataset = create_pseudo_labeled_dataset(
                    self.public_dataset,
                    consensus_labels,
                )
                trainloader = combine_with_pseudo_labels(
                    self.trainloader,
                    self.pseudo_dataset,
                    self.public_batch_size,
                )

            optimizer = config.get("optimizer", "SGD")
            learning_rate = float(config.get("learning_rate", 0.01))

            train_loss, train_acc = train(
                self.net,
                trainloader,
                epochs=1,
                device=self.device,
                optimizer_name=optimizer,
                learning_rate=learning_rate,
                public_loader=None,
            )
            test_loss, test_acc = test(self.net, self.valloader, self.device)

            metrics = {
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "predictions": "",
            }

            if bool(config.get("make_predictions", False)):
                predictions = predict_on_unlabeled(
                    self.net,
                    self.public_dataset,
                    self.device,
                )
                metrics["predictions"] = ",".join(map(str, predictions))

            self._save_state()
            return [], len(self.trainloader.dataset), metrics

        except Exception as exc:
            import traceback

            print(f"[Client {self.partition_id + 1}] ERROR in fit: {exc}")
            print(traceback.format_exc())
            raise

    def evaluate(self, parameters, config):
        self._ensure_loaded()
        test_loss, test_acc = test(self.net, self.valloader, self.device)
        return float(test_loss), len(self.valloader.dataset), {
            "test_acc": float(test_acc)
        }


def client_fn(context: Context):
    """Factory function to create FedCT Flower clients."""
    num_partitions = int(context.run_config.get("num-clients", 5))
    unlabeled_size = int(context.run_config.get("unlabeled-size", 100))
    dataset = context.run_config.get("dataset", "CIFAR10")
    architecture = context.run_config.get("architecture", "ResNet18")
    private_batch_size = int(context.run_config.get("private-batch-size", 32))
    public_batch_size = int(context.run_config.get("public-batch-size", 32))

    partition_id = int(
        context.node_config.get(
            "partition-id",
            context.node_config.get("partition_id", context.node_id % num_partitions),
        )
    )

    return FedCTClient(
        partition_id=partition_id,
        num_partitions=num_partitions,
        unlabeled_size=unlabeled_size,
        dataset=dataset,
        architecture=architecture,
        private_batch_size=private_batch_size,
        public_batch_size=public_batch_size,
        state=context.state,
    ).to_client()


app = ClientApp(client_fn=client_fn)
