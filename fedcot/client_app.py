"""
Flower Client App for FedCT with State Persistence
Handles client-side training, prediction, and model checkpointing
"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import torch
import os
from pathlib import Path

from fedcot.task import (
    get_model,
    load_data,
    train,
    test,
    predict_on_unlabeled,
    create_pseudo_labeled_dataset,
    combine_with_pseudo_labels,
)


class FedCTClient(NumPyClient):
    """
    FedCT Flower Client with State Persistence

    Implements Algorithm 1 from the paper:
    - Local training on Di ∪ P^t
    - Prediction on public dataset U
    - Returns hard labels (not model weights)
    - Saves/loads model checkpoints
    """

    def __init__(self, partition_id: int, num_partitions: int, unlabeled_size: int = 500,
                 dataset: str = "CIFAR10", private_batch_size: int = 32, public_batch_size: int = 32,
                 data_distribution: str = "iid", alpha: float = 0.5, shards_per_client: int = 2,
                 save_checkpoints: bool = True, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize client

        Args:
            partition_id: ID of this client (0-indexed)
            num_partitions: Total number of clients
            unlabeled_size: Size of public unlabeled dataset
            dataset: Dataset to use - "CIFAR10" or "FashionMNIST"
            private_batch_size: Batch size for private data
            public_batch_size: Batch size for public data
            data_distribution: "iid", "non-iid-dirichlet", or "non-iid-shards"
            alpha: Dirichlet concentration parameter
            shards_per_client: Number of shards per client (for non-iid-shards)
            save_checkpoints: Whether to save model checkpoints
            checkpoint_dir: Directory to save checkpoints
        """
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.unlabeled_size = unlabeled_size
        self.dataset = dataset
        self.private_batch_size = private_batch_size
        self.public_batch_size = public_batch_size
        self.data_distribution = data_distribution
        self.alpha = alpha
        self.shards_per_client = shards_per_client
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = Path(checkpoint_dir)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✓ Using GPU: {gpu_name}")
            except:
                print("✓ Using GPU: CUDA device detected")
        else:
            print("⚠️  WARNING: CUDA not available, running on CPU")

        # Data loaders (will be initialized on first fit() call)
        self.trainloader = None
        self.valloader = None
        self.public_dataset = None
        self.pseudo_dataset = None
        self.net = None
        self.data_loaded = False

        # State tracking
        self.current_round = 0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }

    def _get_checkpoint_path(self, round_num: int = None):
        """Get checkpoint file path for this client"""
        if round_num is None:
            round_num = self.current_round
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir / f"client_{self.partition_id}_round_{round_num}.pt"

    def save_state(self, round_num: int = None):
        """
        Save client state (model + training history)

        Args:
            round_num: Round number for checkpoint naming
        """
        if not self.save_checkpoints or self.net is None:
            return

        checkpoint_path = self._get_checkpoint_path(round_num)

        state = {
            'round': self.current_round,
            'model_state_dict': self.net.state_dict(),
            'training_history': self.training_history,
            'partition_id': self.partition_id,
            'dataset': self.dataset,
        }

        torch.save(state, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")

    def load_state(self, round_num: int = None):
        """
        Load client state from checkpoint

        Args:
            round_num: Round number to load (None = latest)

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not self.save_checkpoints:
            return False

        # Find checkpoint
        if round_num is not None:
            checkpoint_path = self._get_checkpoint_path(round_num)
        else:
            # Find latest checkpoint
            checkpoints = list(self.checkpoint_dir.glob(f"client_{self.partition_id}_round_*.pt"))
            if not checkpoints:
                return False
            checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))

        if not checkpoint_path.exists():
            return False

        try:
            state = torch.load(checkpoint_path, map_location=self.device)

            if self.net is not None:
                self.net.load_state_dict(state['model_state_dict'])

            self.current_round = state['round']
            self.training_history = state['training_history']

            print(f"📂 Loaded checkpoint from round {self.current_round}: {checkpoint_path}")
            return True

        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            return False

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
            # Initialize data on first fit() call
            if not self.data_loaded:
                logical_partition_id = config.get("logical_partition_id", self.partition_id)

                # Load data with distribution type
                self.trainloader, self.valloader, self.public_dataset = load_data(
                    partition_id=logical_partition_id,
                    num_partitions=self.num_partitions,
                    unlabeled_size=self.unlabeled_size,
                    dataset=self.dataset,
                    batch_size=self.private_batch_size,
                    data_distribution=self.data_distribution,
                    alpha=self.alpha,
                    shards_per_client=self.shards_per_client
                )

                # Create model
                self.net = get_model(dataset=self.dataset)

                # Try to load previous checkpoint
                self.load_state()

                self.data_loaded = True
                print(f"✓ Client {logical_partition_id + 1} initialized on {self.device}")

            # Increment round counter
            self.current_round += 1

            # Update pseudo-labeled dataset if consensus labels are provided
            public_loader = None
            if "consensus_labels" in config:
                consensus_labels = [int(x) for x in config["consensus_labels"].split(",")]
                self.pseudo_dataset = create_pseudo_labeled_dataset(
                    self.public_dataset, consensus_labels
                )
                _, public_loader = combine_with_pseudo_labels(
                    self.trainloader, self.pseudo_dataset, self.public_batch_size
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
                public_loader=public_loader
            )

            # Check if we need to make predictions
            make_predictions = config.get("make_predictions", False)

            metrics = {"predictions": ""}

            if make_predictions:
                # Evaluate and share metrics at communication rounds
                test_loss, test_acc = test(self.net, self.valloader, self.device)
                predictions = predict_on_unlabeled(self.net, self.public_dataset, self.device)
                predictions_str = ",".join(map(str, predictions))

                # Update history
                self.training_history['train_loss'].append(float(train_loss))
                self.training_history['train_acc'].append(float(train_acc))
                self.training_history['test_loss'].append(float(test_loss))
                self.training_history['test_acc'].append(float(test_acc))

                metrics = {
                    "predictions": predictions_str,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "test_loss": float(test_loss),
                    "test_acc": float(test_acc),
                }

                # Save checkpoint at communication rounds
                self.save_state()

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

    Args:
        context: Flower context with node configuration

    Returns:
        FedCTClient instance
    """
    # Get configuration from run config
    num_partitions = context.run_config.get("num-clients", 5)
    unlabeled_size = context.run_config.get("unlabeled-size", 100)
    dataset = context.run_config.get("dataset", "CIFAR10")
    private_batch_size = context.run_config.get("private-batch-size", 32)
    public_batch_size = context.run_config.get("public-batch-size", 32)
    data_distribution = context.run_config.get("data-distribution", "iid")
    alpha = context.run_config.get("alpha", 0.5)
    shards_per_client = context.run_config.get("shards-per-client", 2)
    save_checkpoints = context.run_config.get("save-checkpoints", True)
    checkpoint_dir = context.run_config.get("checkpoint-dir", "./checkpoints")

    return FedCTClient(
        partition_id=0,  # Will be overridden by logical_partition_id
        num_partitions=num_partitions,
        unlabeled_size=unlabeled_size,
        dataset=dataset,
        private_batch_size=private_batch_size,
        public_batch_size=public_batch_size,
        data_distribution=data_distribution,
        alpha=alpha,
        shards_per_client=shards_per_client,
        save_checkpoints=save_checkpoints,
        checkpoint_dir=checkpoint_dir
    ).to_client()


# Create Flower ClientApp
app = ClientApp(client_fn=client_fn)