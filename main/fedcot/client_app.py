"""
Flower Client App for FedCT
Handles client-side training and prediction
"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import torch
import logging

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


class FedCTClient(NumPyClient):
    """
    FedCT Flower Client

    Implements Algorithm 1 from the paper:
    - Local training on Di ∪ P^t
    - Prediction on public dataset U
    - Returns hard labels (not model weights)
    """

    def __init__(self, partition_id: int, num_partitions: int, unlabeled_size: int = 500, 
                 dataset: str = "CIFAR10", private_batch_size: int = 32, public_batch_size: int = 32):
        """
        Initialize client

        Args:
            partition_id: ID of this client (0-indexed)
            num_partitions: Total number of clients
            unlabeled_size: Size of public unlabeled dataset
            dataset: Dataset to use - "CIFAR10" or "FashionMNIST"
            private_batch_size: Batch size for private data
            public_batch_size: Batch size for public data
        """
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.unlabeled_size = unlabeled_size
        self.dataset = dataset
        self.private_batch_size = private_batch_size
        self.public_batch_size = public_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Data loaders (will be initialized on first fit() call)
        self.trainloader = None
        self.valloader = None
        self.public_dataset = None
        self.pseudo_dataset = None
        self.net = None
        self.data_loaded = False

    def fit(self, parameters, config):
        """
        Train the client model for one local round

        Args:
            parameters: Model parameters (not used in FedCT)
            config: Configuration dict

        Returns:
            Empty parameters, dataset size, and metrics
        """
        try:
            # Initialize data on first fit() call using logical_partition_id from config
            if not self.data_loaded:
                logical_partition_id = config.get("logical_partition_id", self.partition_id)
                
                # Load data
                self.trainloader, self.valloader, self.public_dataset = load_data(
                    partition_id=logical_partition_id,
                    num_partitions=self.num_partitions,
                    unlabeled_size=self.unlabeled_size,
                    dataset=self.dataset,
                    batch_size=self.private_batch_size
                )

                # Create model
                self.net = get_model(dataset=self.dataset)
                
                self.data_loaded = True
                # Only log initialization once (suppress Ray actor repeated messages)
                # logger.info(f"✓ Client {logical_partition_id + 1} initialized on {self.device}")
            
            # Update pseudo-labeled dataset if consensus labels are provided
            public_loader = None
            if "consensus_labels" in config:
                consensus_labels = [int(x) for x in config["consensus_labels"].split(",")]
                self.pseudo_dataset = create_pseudo_labeled_dataset(
                    self.public_dataset, consensus_labels
                )
                # Create separate loader for public data with its own batch size
                _, public_loader = combine_with_pseudo_labels(
                    self.trainloader, self.pseudo_dataset, self.public_batch_size
                )

            # Get training parameters from config
            optimizer = config.get("optimizer", "SGD")
            learning_rate = float(config.get("learning_rate", 0.01))

            # Train model for 1 epoch (1 local round = 1 pass through dataset)
            train_loss, train_acc = train(
                self.net, 
                self.trainloader,
                epochs=1,  # Just 1 epoch per local round
                device=self.device,
                optimizer_name=optimizer,
                learning_rate=learning_rate,
                public_loader=public_loader
            )

            # Always evaluate on test set to get metrics at every local round
            test_loss, test_acc = test(self.net, self.valloader, self.device)
            
            # Check if we need to make predictions (communication round)
            make_predictions = config.get("make_predictions", False)
            
            # Always send train/test metrics, but only send predictions at communication rounds
            metrics = {
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "predictions": ""
            }
            
            if make_predictions:
                # At communication rounds, also make predictions on public dataset
                predictions = predict_on_unlabeled(self.net, self.public_dataset, self.device)
                predictions_str = ",".join(map(str, predictions))
                metrics["predictions"] = predictions_str

            # Return empty parameters (FedCT doesn't share model weights)
            return [], len(self.trainloader.dataset), metrics
            
        except Exception as e:
            import traceback
            print(f"[Client {self.partition_id + 1}] ERROR in fit: {e}")
            print(traceback.format_exc())
            raise

    def evaluate(self, parameters, config):
        """
        Evaluate model on test set

        Args:
            parameters: Model parameters (not used)
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
    Factory function to create client instances

    Called by Flower for each client

    Args:
        context: Flower context with node configuration

    Returns:
        FedCTClient instance wrapped as Flower client
    """
    # Get configuration from run config
    num_partitions = context.run_config.get("num-clients", 5)
    unlabeled_size = context.run_config.get("unlabeled-size", 100)
    dataset = context.run_config.get("dataset", "CIFAR10")
    private_batch_size = context.run_config.get("private-batch-size", 32)
    public_batch_size = context.run_config.get("public-batch-size", 32)

    # Note: partition_id will be set dynamically from config during fit()
    # This is a workaround since we need to support num-supernodes > num-clients
    return FedCTClient(
        partition_id=0,  # Will be overridden by logical_partition_id from config
        num_partitions=num_partitions,
        unlabeled_size=unlabeled_size,
        dataset=dataset,
        private_batch_size=private_batch_size,
        public_batch_size=public_batch_size
    ).to_client()


# Create Flower ClientApp
app = ClientApp(client_fn=client_fn)