"""
Flower Server App for FedCT
Handles server-side aggregation using majority voting
"""

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.server.strategy import Strategy
from flwr.common import Context
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple, Union
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
import logging
import sys

# Configure logging to ensure output is visible
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)


class FedCTStrategy(Strategy):
    """
    FedCT Strategy with Majority Voting

    Implements Algorithm 1, Lines 11-12:
    - Line 11: L̄_t ← consensus(L¹_t, ..., L^m_t)
    - Line 12: send L̄_t to all clients
    """

    def __init__(self, num_clients: int = 5, num_local_rounds: int = 2,
                 optimizer: str = "SGD", learning_rate: float = 0.01, num_communication_rounds: int = 10):
        """
        Initialize strategy

        Args:
            num_clients: Number of clients participating
            num_local_rounds: Number of local rounds before communication
            optimizer: Optimizer to use
            learning_rate: Learning rate
            num_communication_rounds: Total number of communication rounds
        """
        super().__init__()
        self.num_clients = num_clients
        self.num_local_rounds = num_local_rounds
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.num_communication_rounds = num_communication_rounds
        self.previous_consensus = None
        self.consensus_history = []
        self.local_round_counter = 0
        self.comm_round_counter = 0

    def initialize_parameters(self, client_manager):
        """
        Initialize parameters (empty for FedCT)

        Returns:
            Empty parameters
        """
        return ndarrays_to_parameters([])

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Configure clients for one local training round

        Args:
            server_round: Current Flower server round number (sequential: 1, 2, 3, ...)
            parameters: Model parameters (empty for FedCT)
            client_manager: Flower client manager

        Returns:
            List of (ClientProxy, FitIns) tuples
        """
        from flwr.common import FitIns

        self.local_round_counter += 1

        # Check if this is the start of a new communication cycle
        if (self.local_round_counter - 1) % self.num_local_rounds == 0:
            self.comm_round_counter += 1
            logger.info(f"\n{'=' * 80}")
            logger.info(
                f"📊 COMMUNICATION CYCLE {self.comm_round_counter}/{self.num_communication_rounds} (Flower Round {server_round})")
            logger.info(f"{'=' * 80}")
            logger.info(f"→ Clients will do {self.num_local_rounds} local training rounds")
            logger.info(f"→ Then share predictions on public dataset")
            logger.info(f"{'=' * 80}")

        # Sample all clients
        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients
        )

        # Prepare configuration for each client
        # Assign logical partition IDs (0 to num_clients-1) to handle case where
        # num-supernodes > num-clients
        config_list = []
        for idx, client in enumerate(clients):
            config = {
                "optimizer": self.optimizer,
                "learning_rate": self.learning_rate,
                "logical_partition_id": idx,  # Logical ID for data partitioning
            }

            # Send consensus labels if available (at start of communication cycle)
            if self.previous_consensus is not None:
                config["consensus_labels"] = ",".join(map(str, self.previous_consensus))

            # Check if this is the last local round before communication
            is_communication_round = (self.local_round_counter % self.num_local_rounds == 0)
            config["make_predictions"] = is_communication_round

            fit_ins = FitIns(parameters=parameters, config=config)
            config_list.append((client, fit_ins))

        return config_list

    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate results and perform majority voting on communication rounds

        Args:
            server_round: Current Flower server round number
            results: List of (ClientProxy, FitRes) tuples
            failures: List of failed clients

        Returns:
            Empty parameters and aggregation statistics
        """

        if not results:
            return None, {}

        stats = {}

        # Check if this is a communication round
        is_communication_round = (self.local_round_counter % self.num_local_rounds == 0)

        # Calculate position within current communication cycle
        local_within_cycle = ((self.local_round_counter - 1) % self.num_local_rounds) + 1

        # Collect metrics from all clients (always available now)
        client_metrics = {}
        for i, (_, fit_res) in enumerate(results):
            client_metrics[i] = {
                "train_loss": fit_res.metrics.get("train_loss", 0.0),
                "train_acc": fit_res.metrics.get("train_acc", 0.0),
                "test_loss": fit_res.metrics.get("test_loss", 0.0),
                "test_acc": fit_res.metrics.get("test_acc", 0.0),
            }

        # Calculate and display average metrics at every local round
        if client_metrics:
            train_accs = [m["train_acc"] for m in client_metrics.values()]
            test_accs = [m["test_acc"] for m in client_metrics.values()]
            train_losses = [m["train_loss"] for m in client_metrics.values()]
            test_losses = [m["test_loss"] for m in client_metrics.values()]

            avg_train_acc = np.mean(train_accs)
            avg_test_acc = np.mean(test_accs)
            avg_train_loss = np.mean(train_losses)
            avg_test_loss = np.mean(test_losses)

            logger.info(f"✓ Local Round {local_within_cycle}/{self.num_local_rounds} [Flower Round {server_round}]: "
                        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                        f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")

        # If this is a communication round, also do majority voting
        if is_communication_round:
            logger.info(f"\n{'=' * 80}")
            logger.info(
                f"🗳️  COMMUNICATION ROUND {self.comm_round_counter}/{self.num_communication_rounds} - MAJORITY VOTING")
            logger.info(f"{'=' * 80}")

            # Extract predictions from all clients
            client_predictions = {}

            for i, (_, fit_res) in enumerate(results):
                predictions_str = fit_res.metrics.get("predictions", "")
                if predictions_str:  # Only process if we have predictions
                    predictions = [int(x) for x in predictions_str.split(",")]
                    client_predictions[i] = predictions

            # Perform majority voting
            if client_predictions:
                consensus_labels, consensus_stats = self._majority_vote(client_predictions)
                self._print_summary(consensus_labels, consensus_stats, client_metrics)

                # Store consensus for next communication cycle
                self.previous_consensus = consensus_labels
                self.consensus_history.append(consensus_labels)

                stats.update(consensus_stats)

        # Return empty parameters (FedCT doesn't aggregate model weights)
        return ndarrays_to_parameters([]), stats

    def _majority_vote(self, client_predictions: Dict[int, List[int]]):
        """
        Perform majority voting on client predictions

        Args:
            client_predictions: Dict mapping client_id to their predictions

        Returns:
            consensus_labels: List of consensus labels
            stats: Dictionary with agreement statistics
        """
        num_clients = len(client_predictions)
        num_samples = len(next(iter(client_predictions.values())))

        consensus_labels = []
        agreements = []
        unanimous_count = 0

        # Vote for each sample
        for sample_idx in range(num_samples):
            votes = [client_predictions[cid][sample_idx]
                     for cid in sorted(client_predictions.keys())]

            # Count votes
            vote_counts = Counter(votes)
            winner = vote_counts.most_common(1)[0][0]
            consensus_labels.append(winner)

            # Calculate agreement
            agreement = vote_counts[winner] / num_clients
            agreements.append(agreement)

            if agreement == 1.0:
                unanimous_count += 1

        # Calculate how many labels changed
        labels_changed = 0
        if self.previous_consensus is not None:
            labels_changed = sum(1 for i in range(num_samples)
                                 if consensus_labels[i] != self.previous_consensus[i])

        stats = {
            'mean_agreement': float(np.mean(agreements)),
            'min_agreement': float(np.min(agreements)),
            'max_agreement': float(np.max(agreements)),
            'unanimous': float(unanimous_count),
            'changed': float(labels_changed),
        }

        return consensus_labels, stats

    def _print_summary(self, consensus, stats, client_metrics):
        """
        Print summary of aggregation

        Args:
            consensus: Consensus labels
            stats: Agreement statistics
            client_metrics: Client training metrics
        """
        num_samples = len(consensus)

        logger.info(f"\n{'-' * 80}")
        logger.info("MAJORITY VOTING RESULTS:")
        logger.info(f"{'-' * 80}")
        logger.info(f"Consensus Labels (L̄_t) - {num_samples} samples:")
        if num_samples <= 20:
            logger.info(f"  {consensus}")
        else:
            logger.info(f"  {consensus[:20]}... (showing first 20)")

        logger.info(f"\nAgreement Statistics:")
        logger.info(f"  Mean Agreement: {stats['mean_agreement']:.3f}")
        logger.info(f"  Min Agreement:  {stats['min_agreement']:.3f}")
        logger.info(f"  Max Agreement:  {stats['max_agreement']:.3f}")
        logger.info(f"  Unanimous:      {int(stats['unanimous'])}/{num_samples}")
        logger.info(f"  Labels Changed: {int(stats['changed'])}/{num_samples}")

        logger.info(f"\nClient Performance Summary:")
        test_accs = [m["test_acc"] for m in client_metrics.values()]
        train_accs = [m["train_acc"] for m in client_metrics.values()]
        test_losses = [m["test_loss"] for m in client_metrics.values()]
        train_losses = [m["train_loss"] for m in client_metrics.values()]

        for cid, metrics in client_metrics.items():
            logger.info(
                f"  Client {cid + 1}: Train Loss={metrics['train_loss']:.4f}, Train Acc={metrics['train_acc']:.4f}, "
                f"Test Loss={metrics['test_loss']:.4f}, Test Acc={metrics['test_acc']:.4f}")

        logger.info(f"\n  Average Train Loss: {np.mean(train_losses):.4f} ± {np.std(train_losses):.4f}")
        logger.info(f"  Average Train Acc:  {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
        logger.info(f"  Average Test Loss:  {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")
        logger.info(f"  Average Test Acc:   {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")

        logger.info(f"{'=' * 80}\n")

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure evaluation (not used in FedCT)"""
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation (not used in FedCT)"""
        return None, {}

    def evaluate(self, server_round, parameters):
        """Server-side evaluation (not used in FedCT)"""
        return None


def server_fn(context: Context):
    """
    Create server components for Flower

    Args:
        context: Flower context with run configuration

    Returns:
        ServerAppComponents with strategy and config
    """
    # Get configuration from run_config
    num_communication_rounds = context.run_config.get("num-communication-rounds", 10)
    num_clients = context.run_config.get("num-clients", 5)
    num_local_rounds = context.run_config.get("num-local-rounds", 2)
    optimizer = context.run_config.get("optimizer", "SGD")
    learning_rate = context.run_config.get("learning-rate", 0.01)
    unlabeled_size = context.run_config.get("unlabeled-size", 100)
    dataset = context.run_config.get("dataset", "CIFAR10")

    # Calculate total Flower server rounds (local rounds × communication rounds)
    total_server_rounds = num_communication_rounds * num_local_rounds

    logger.info(f"\n{'=' * 80}")
    logger.info("FEDERATED CO-TRAINING (FedCT) - FLOWER IMPLEMENTATION")
    logger.info(f"{'=' * 80}")
    logger.info(f"Configuration:")
    logger.info(f"  Dataset:                       {dataset}")
    logger.info(f"  Num communication rounds:      {num_communication_rounds}")
    logger.info(f"  Number of clients:             {num_clients}")
    logger.info(f"  Num local rounds per cycle:    {num_local_rounds}")
    logger.info(f"  Total Flower server rounds:    {total_server_rounds}")
    logger.info(f"  Optimizer:                     {optimizer}")
    logger.info(f"  Learning rate:                 {learning_rate}")
    logger.info(f"  Unlabeled size:                {unlabeled_size}")
    logger.info(f"\n  Note: Flower counts rounds sequentially (1,2,3,...)")
    logger.info(f"        We organize them into {num_communication_rounds} communication cycles")
    logger.info(f"        Each cycle has {num_local_rounds} local training rounds")
    logger.info(f"{'=' * 80}\n")

    # Create strategy
    strategy = FedCTStrategy(
        num_clients=num_clients,
        num_local_rounds=num_local_rounds,
        optimizer=optimizer,
        learning_rate=learning_rate,
        num_communication_rounds=num_communication_rounds
    )

    # Create server config with total rounds
    config = ServerConfig(num_rounds=total_server_rounds, round_timeout=None)

    return ServerAppComponents(strategy=strategy, config=config)


# Create Flower ServerApp
app = ServerApp(server_fn=server_fn)