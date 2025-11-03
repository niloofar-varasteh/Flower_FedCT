"""
Flower Client App for FedCT with State Persistence
Handles client-side training, prediction, and model checkpointing
"""

from pathlib import Path
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import torch

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

    def __init__(
        self,
        partition_id: int,
        num_partitions: int,
        unlabeled_size: int = 500,
        dataset: str = "CIFAR10",
        private_batch_size: int = 32,
        public_batch_size: int = 32,
        data_distribution: str = "iid",
        alpha: float = 0.5,
        shards_per_client: int = 2,
        save_checkpoints: bool = True,
        checkpoint_dir: str = "./checkpoints",
    ):
        # IDs & data config
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.unlabeled_size = unlabeled_size
        self.dataset = dataset
        self.private_batch_size = private_batch_size
        self.public_batch_size = public_batch_size
        self.data_distribution = data_distribution
        self.alpha = alpha
        self.shards_per_client = shards_per_client

        # Checkpoints
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = Path(checkpoint_dir)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✓ Using GPU: {gpu_name}")
            except Exception:
                print("✓ Using GPU: CUDA device detected")
        else:
            print("⚠️  WARNING: CUDA not available, running on CPU")

        # Runtime state
        self.trainloader = None
        self.valloader = None
        self.public_dataset = None
        self.pseudo_dataset = None
        self.net = None
        self.data_loaded = False

        self.current_round = 0
        self.training_history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

    # ---------------------------
    # Checkpoint helpers
    # ---------------------------
    def _get_checkpoint_path(self, round_num: int | None = None) -> Path:
        if round_num is None:
            round_num = self.current_round
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir / f"client_{self.partition_id}_round_{round_num}.pt"

    def save_state(self, round_num: int | None = None) -> None:
        if not self.save_checkpoints or self.net is None:
            return
        checkpoint_path = self._get_checkpoint_path(round_num)
        state = {
            "round": self.current_round,
            "model_state_dict": self.net.state_dict(),
            "training_history": self.training_history,
            "partition_id": self.partition_id,
            "dataset": self.dataset,
        }
        torch.save(state, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")

    def load_state(self, round_num: int | None = None) -> bool:
        if not self.save_checkpoints:
            return False

        if round_num is not None:
            checkpoint_path = self._get_checkpoint_path(round_num)
        else:
            checkpoints = list(self.checkpoint_dir.glob(f"client_{self.partition_id}_round_*.pt"))
            if not checkpoints:
                return False
            checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))

        if not checkpoint_path.exists():
            return False

        try:
            state = torch.load(checkpoint_path, map_location=self.device)
            if self.net is not None:
                self.net.load_state_dict(state["model_state_dict"])
            self.current_round = state["round"]
            self.training_history = state["training_history"]
            print(f"📂 Loaded checkpoint from round {self.current_round}: {checkpoint_path}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            return False

    # ---------------------------
    # Flower NumPyClient API
    # ---------------------------
    def fit(self, parameters, config):
        """
        Train the client model for one local round.
        FedCT returns predictions rather than model weights.
        """
        try:
            # Initialize data/model once
            if not self.data_loaded:
                raw_pid = config.get("logical_partition_id", self.partition_id)
                try:
                    logical_partition_id = int(raw_pid)
                except (TypeError, ValueError):
                    logical_partition_id = 0  # safe fallback

                # DEBUG
                print(f"\n🔧 DEBUG Client {logical_partition_id + 1}:")
                print(f"   data_distribution = {self.data_distribution}")
                print(f"   alpha = {self.alpha}")
                print(f"   shards_per_client = {self.shards_per_client}")

                # Load data with proper kwargs (NO Ellipsis)
                self.trainloader, self.valloader, self.public_dataset = load_data(
                    partition_id=logical_partition_id,
                    num_partitions=self.num_partitions,
                    unlabeled_size=self.unlabeled_size,
                    dataset=self.dataset,  # if task.py expects dataset_name, rename here
                    batch_size=self.private_batch_size,
                    data_distribution=self.data_distribution,
                    alpha=self.alpha,
                    shards_per_client=self.shards_per_client,
                )

                # Create model and attempt to restore state
                self.net = get_model(dataset=self.dataset)
                self.load_state()

                self.data_loaded = True
                print(f"✓ Client {logical_partition_id + 1} initialized on {self.device}")

            # Increment local round
            self.current_round += 1

            # Pseudo-label updates (if server provided consensus)
            public_loader = None
            if "consensus_labels" in config:
                raw = str(config["consensus_labels"]).strip()
                if raw:
                    consensus_labels = [int(x) for x in raw.split(",")]
                    self.pseudo_dataset = create_pseudo_labeled_dataset(
                        self.public_dataset, consensus_labels
                    )
                    _, public_loader = combine_with_pseudo_labels(
                        self.trainloader, self.pseudo_dataset, self.public_batch_size
                    )

            # Optimizer config
            optimizer = config.get("optimizer", "SGD")
            learning_rate = float(config.get("learning_rate", 0.01))

            # Train one epoch
            train_loss, train_acc = train(
                self.net,
                self.trainloader,
                epochs=1,
                device=self.device,
                optimizer_name=optimizer,
                learning_rate=learning_rate,
                public_loader=public_loader,
            )

            # Default metrics
            metrics = {"predictions": ""}

            # At communication rounds: evaluate + predict on U
            make_predictions = bool(config.get("make_predictions", False))
            if make_predictions:
                test_loss, test_acc = test(self.net, self.valloader, self.device)
                predictions = predict_on_unlabeled(self.net, self.public_dataset, self.device)
                predictions_str = ",".join(map(str, predictions))

                # History
                self.training_history["train_loss"].append(float(train_loss))
                self.training_history["train_acc"].append(float(train_acc))
                self.training_history["test_loss"].append(float(test_loss))
                self.training_history["test_acc"].append(float(test_acc))

                metrics = {
                    "predictions": predictions_str,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "test_loss": float(test_loss),
                    "test_acc": float(test_acc),
                }

                # Save at comm rounds
                self.save_state()

            # FedCT returns empty weights
            return [], len(self.trainloader.dataset), metrics

        except Exception as e:
            import traceback
            print(f"[Client {self.partition_id + 1}] ERROR in fit: {e}")
            print(traceback.format_exc())
            raise

    def evaluate(self, parameters, config):
        """Evaluate on the validation set."""
        test_loss, test_acc = test(self.net, self.valloader, self.device)
        return float(test_loss), len(self.valloader.dataset), {"test_acc": float(test_acc)}


# ---------------------------
# Client factory
# ---------------------------
def _rc(rc, key, default=None):
    # support both: kebab-case و snake_case is just a utility for reading configuration values from Flower’s context.run_config dictionary (the rc variable).
    return rc.get(key, rc.get(key.replace("-", "_"), default))


def client_fn(context: Context):
    rc = context.run_config




    num_partitions = int(_rc(rc, "num-clients", 5))
    unlabeled_size = int(_rc(rc, "unlabeled-size", 100))
    dataset = _rc(rc, "dataset", "CIFAR10")
    private_batch = int(_rc(rc, "private-batch-size", 32))
    public_batch = int(_rc(rc, "public-batch-size", 32))
    data_distribution = _rc(rc, "data-distribution", "iid")
    alpha = _rc(rc, "alpha", None)
    alpha = float(alpha) if alpha is not None else None
    # NEW: if IID, alpha is not used at all
    if str(data_distribution).lower() == "iid":
        alpha = None
    shards_per_client = int(_rc(rc, "shards-per-client", 2))
    save_checkpoints = bool(_rc(rc, "save-checkpoints", True))
    checkpoint_dir = _rc(rc, "checkpoint-dir", "./checkpoints")

    return FedCTClient(
        partition_id=0,  # logical_partition_id is supplied via config in fit()
        num_partitions=num_partitions,
        unlabeled_size=unlabeled_size,
        dataset=dataset,
        private_batch_size=private_batch,
        public_batch_size=public_batch,
        data_distribution=data_distribution,
        alpha=alpha,  # NEW: do not force a default; None means “unused for IID”
        shards_per_client=shards_per_client,
        save_checkpoints=save_checkpoints,
        checkpoint_dir=checkpoint_dir,
    ).to_client()


# Create Flower ClientApp
app = ClientApp(client_fn=client_fn)
