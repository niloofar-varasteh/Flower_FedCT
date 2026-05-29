"""
Flower Client App for FedCT
Handles client-side training and prediction
"""

import logging

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from fedcot.task import (
    get_model,
    load_data,
    train,
    test,
    predict_on_unlabeled,
    create_pseudo_labeled_dataset,
    combine_with_pseudo_labels,
)

logger = logging.getLogger(__name__)


def normalize_architecture(arch: str) -> str:
    """Normalize architecture name."""
    if arch.lower() == "lightcnn":
        return "LightCNN"
    return "ResNet18"


class FedCTClient(NumPyClient):
    """
    FedCT Flower Client

    Implements Algorithm 1 from the paper:
    - Local training on Di ∪ P^t
    - Prediction on public dataset U
    - Returns hard labels, not model weights
    """

    def __init__(
        self,
        partition_id: int,
        num_partitions: int,
        unlabeled_size: int = 500,
        dataset: str = "CIFAR10",
        architecture: str = "ResNet18",
        private_batch_size: int = 32,
        public_batch_size: int = 32,
    ):
        """
        Initialize client.

        Args:
            partition_id: ID of this client
            num_partitions: Total number of clients
            unlabeled_size: Size of public unlabeled dataset
            dataset: Dataset to use - "CIFAR10" or "FashionMNIST"
            architecture: Model architecture - "ResNet18" or "LightCNN"
            private_batch_size: Batch size for private data
            public_batch_size: Batch size for public data
        """
        self.partition_id = partition_id
        self.num_partitions = int(num_partitions)
        self.unlabeled_size = int(unlabeled_size)
        self.dataset = dataset
        self.architecture = normalize_architecture(architecture)
        self.private_batch_size = int(private_batch_size)
        self.public_batch_size = int(public_batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data loaders will be initialized on first fit call
        self.trainloader = None
        self.valloader = None
        self.public_dataset = None
        self.pseudo_dataset = None
        self.net = None
        self.data_loaded = False

    def fit(self, parameters, config):
        """
        Train the client model for one local round.

        Args:
            parameters: Model parameters, not used in FedCT
            config: Configuration dict

        Returns:
            Empty parameters, dataset size, and metrics
        """
        try:
            # Initialize data on first fit call
            if not self.data_loaded:
                logical_partition_id = int(
                    config.get("logical_partition_id", self.partition_id)
                )

                self.trainloader, self.valloader, self.public_dataset = load_data(
                    partition_id=logical_partition_id,
                    num_partitions=self.num_partitions,
                    unlabeled_size=self.unlabeled_size,
                    dataset=self.dataset,
                    batch_size=self.private_batch_size,
                )

                self.net = get_model(
                    dataset=self.dataset,
                    architecture=self.architecture,
                )

                self.data_loaded = True

            # Update pseudo-labeled dataset if consensus labels are provided
            public_loader = None
            if "consensus_labels" in config:
                consensus_labels = [
                    int(x) for x in config["consensus_labels"].split(",") if x != ""
                ]

                self.pseudo_dataset = create_pseudo_labeled_dataset(
                    self.public_dataset,
                    consensus_labels,
                )

                _, public_loader = combine_with_pseudo_labels(
                    self.trainloader,
                    self.pseudo_dataset,
                    self.public_batch_size,
                )

            # Get training parameters from config
            optimizer = config.get("optimizer", "SGD")
            learning_rate = float(config.get("learning_rate", 0.01))

            # Train model for 1 epoch
            train_loss, train_acc = train(
                self.net,
                self.trainloader,
                epochs=1,
                device=self.device,
                optimizer_name=optimizer,
                learning_rate=learning_rate,
                public_loader=public_loader,
            )

            # Evaluate on test set
            test_loss, test_acc = test(self.net, self.valloader, self.device)

            # Check if predictions are needed for communication round
            make_predictions = bool(config.get("make_predictions", False))

            metrics = {
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "predictions": "",
            }

            if make_predictions:
                predictions = predict_on_unlabeled(
                    self.net,
                    self.public_dataset,
                    self.device,
                )
                metrics["predictions"] = ",".join(map(str, predictions))

            # Return empty parameters because FedCT does not share model weights
            return [], len(self.trainloader.dataset), metrics

        except Exception as e:
            import traceback

            print(f"[Client {self.partition_id + 1}] ERROR in fit: {e}")
            print(traceback.format_exc())
            raise

    def evaluate(self, parameters, config):
        """
        Evaluate model on test set.

        Args:
            parameters: Model parameters, not used
            config: Configuration dict

        Returns:
            Test loss, dataset size, and metrics
        """
        test_loss, test_acc = test(self.net, self.valloader, self.device)

        return float(test_loss), len(self.valloader.dataset), {
            "test_acc": float(test_acc)
        }


def client_fn(context: Context):
    """
    Factory function to create client instances.

    Args:
        context: Flower context with node configuration

    Returns:
        FedCTClient instance wrapped as Flower client
    """
    num_partitions = context.run_config.get("num-clients", 5)
    unlabeled_size = context.run_config.get("unlabeled-size", 100)
    dataset = context.run_config.get("dataset", "CIFAR10")
    architecture = context.run_config.get("architecture", "ResNet18")
    private_batch_size = context.run_config.get("private-batch-size", 32)
    public_batch_size = context.run_config.get("public-batch-size", 32)

    return FedCTClient(
        partition_id=0,
        num_partitions=int(num_partitions),
        unlabeled_size=int(unlabeled_size),
        dataset=dataset,
        architecture=architecture,
        private_batch_size=int(private_batch_size),
        public_batch_size=int(public_batch_size),
    ).to_client()


# Create Flower ClientApp
app = ClientApp(client_fn=client_fn)