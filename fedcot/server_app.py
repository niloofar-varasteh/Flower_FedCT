"""
Flower Server App for FedDCT
Handles server-side aggregation using majority voting
"""

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import Strategy
from flwr.common import Context
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple, Union
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy


class FedDCTStrategy(Strategy):
    """
    FedDCT Strategy with Majority Voting

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
        Initialize parameters (empty for FedDCT)

        Returns:
            Empty parameters
        """
        return ndarrays_to_parameters([])

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Configure clients for one local training round

        Args:
            server_round: Current Flower server round number
            parameters: Model parameters (empty for FedDCT)
            client_manager: Flower client manager

        Returns:
            List of (ClientProxy, FitIns) tuples
        """
        from flwr.common import FitIns

        self.local_round_counter += 1
        
        # Check if this is the start of a new communication cycle
        if (self.local_round_counter - 1) % self.num_local_rounds == 0:
            self.comm_round_counter += 1
            print(f"\n{'=' * 80}")
            print(f"📊 COMMUNICATION CYCLE {self.comm_round_counter}/{self.num_communication_rounds}")
            print(f"{'=' * 80}")
            print(f"→ Clients will do {self.num_local_rounds} local training rounds")
            print(f"→ Then share predictions on public dataset")
            print(f"{'=' * 80}")

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
        
        # Always print local round completion message
        local_within_cycle = ((self.local_round_counter - 1) % self.num_local_rounds) + 1
        print(f"✓ Local Round {local_within_cycle}/{self.num_local_rounds}: Training completed (metrics shared only at communication rounds)")
        
        # If this is a communication round, also do majority voting
        if is_communication_round:
            print(f"\n{'=' * 80}")
            print(f"🗳️  COMMUNICATION ROUND {self.comm_round_counter}/{self.num_communication_rounds} - MAJORITY VOTING")
            print(f"{'=' * 80}")

            # Extract predictions and metrics from all clients
            client_predictions = {}
            client_metrics = {}

            for i, (_, fit_res) in enumerate(results):
                predictions_str = fit_res.metrics.get("predictions", "")
                if predictions_str:  # Only process if we have predictions
                    predictions = [int(x) for x in predictions_str.split(",")]
                    client_predictions[i] = predictions
                    client_metrics[i] = {
                        "train_loss": fit_res.metrics.get("train_loss", 0.0),
                        "train_acc": fit_res.metrics.get("train_acc", 0.0),
                        "test_loss": fit_res.metrics.get("test_loss", 0.0),
                        "test_acc": fit_res.metrics.get("test_acc", 0.0),
                    }

            # Perform majority voting
            if client_predictions:
                consensus_labels, consensus_stats = self._majority_vote(client_predictions)
                self._print_summary(consensus_labels, consensus_stats, client_metrics)
                
                # Store consensus for next communication cycle
                self.previous_consensus = consensus_labels
                self.consensus_history.append(consensus_labels)
                
                stats.update(consensus_stats)

        # Return empty parameters (FedDCT doesn't aggregate model weights)
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
        
        print(f"\n{'-' * 80}")
        print("MAJORITY VOTING RESULTS:")
        print(f"{'-' * 80}")
        print(f"Consensus Labels (L̄_t) - {num_samples} samples:")
        if num_samples <= 20:
            print(f"  {consensus}")
        else:
            print(f"  {consensus[:20]}... (showing first 20)")

        print(f"\nAgreement Statistics:")
        print(f"  Mean Agreement: {stats['mean_agreement']:.3f}")
        print(f"  Min Agreement:  {stats['min_agreement']:.3f}")
        print(f"  Max Agreement:  {stats['max_agreement']:.3f}")
        print(f"  Unanimous:      {int(stats['unanimous'])}/{num_samples}")
        print(f"  Labels Changed: {int(stats['changed'])}/{num_samples}")

        print(f"\nClient Performance Summary:")
        test_accs = [m["test_acc"] for m in client_metrics.values()]
        train_accs = [m["train_acc"] for m in client_metrics.values()]
        test_losses = [m["test_loss"] for m in client_metrics.values()]
        train_losses = [m["train_loss"] for m in client_metrics.values()]
        
        for cid, metrics in client_metrics.items():
            print(f"  Client {cid + 1}: Train Loss={metrics['train_loss']:.4f}, Train Acc={metrics['train_acc']:.4f}, "
                  f"Test Loss={metrics['test_loss']:.4f}, Test Acc={metrics['test_acc']:.4f}")
        
        print(f"\n  Average Train Loss: {np.mean(train_losses):.4f} ± {np.std(train_losses):.4f}")
        print(f"  Average Train Acc:  {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
        print(f"  Average Test Loss:  {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")
        print(f"  Average Test Acc:   {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")

        print(f"{'=' * 80}\n")

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure evaluation (not used in FedDCT)"""
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation (not used in FedDCT)"""
        return None, {}

    def evaluate(self, server_round, parameters):
        """Server-side evaluation (not used in FedDCT)"""
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
    
    print(f"\n{'=' * 80}")
    print("FEDERATED CO-TRAINING (FedDCT) - FLOWER IMPLEMENTATION")
    print(f"{'=' * 80}")
    print(f"Configuration:")
    print(f"  Dataset:                       {dataset}")
    print(f"  Num communication rounds:      {num_communication_rounds}")
    print(f"  Number of clients:             {num_clients}")
    print(f"  Num local rounds:              {num_local_rounds}")
    print(f"  Total Flower server rounds:    {total_server_rounds}")
    print(f"  Optimizer:                     {optimizer}")
    print(f"  Learning rate:                 {learning_rate}")
    print(f"  Unlabeled size:                {unlabeled_size}")
    print(f"{'=' * 80}\n")
    
    # Create strategy
    strategy = FedDCTStrategy(
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