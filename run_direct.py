#!/usr/bin/env python3
"""
FIXED Direct execution script for FedCT with proper model aggregation.
"""

import argparse
import sys
from pathlib import Path

# Add the project to the path
sys.path.insert(0, str(Path(__file__).parent))

from fedcot.task import get_model, load_data, train, test, predict_on_unlabeled, create_pseudo_labeled_dataset, \
    combine_with_pseudo_labels
from collections import Counter
import torch
import numpy as np


def get_state_dict(model):
    """Get model state dict"""
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def set_state_dict(model, state_dict):
    """Set model state dict"""
    model.load_state_dict(state_dict)


def average_state_dicts(state_dicts):
    """Average multiple state dicts (FedAvg aggregation)"""
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.stack([sd[key] for sd in state_dicts]).mean(dim=0)
    return avg_state


def run_fedcot_experiment(num_communication_rounds=15, num_clients=5, num_local_rounds=3, unlabeled_size=100,
                          dataset="CIFAR10", optimizer="Adam", learning_rate=0.001,
                          private_batch_size=64, public_batch_size=32):
    """
    Run FedCT experiment with PROPER model aggregation
    """

    print("=" * 80)
    print("FEDERATED CO-TRAINING (FedCT) - FIXED VERSION WITH MODEL AGGREGATION")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Dataset:                       {dataset}")
    print(f"  Num communication rounds:      {num_communication_rounds}")
    print(f"  Number of clients:             {num_clients}")
    print(f"  Num local rounds:              {num_local_rounds}")
    print(f"  Private batch size:            {private_batch_size}")
    print(f"  Public batch size:             {public_batch_size}")
    print(f"  Optimizer:                     {optimizer}")
    print(f"  Learning rate:                 {learning_rate}")
    print(f"  Unlabeled size:                {unlabeled_size}")
    print("=" * 80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    # Initialize global model
    global_model = get_model(dataset=dataset)

    # Initialize clients
    clients = []
    for client_id in range(num_clients):
        trainloader, valloader, public_dataset = load_data(
            partition_id=client_id,
            num_partitions=num_clients,
            unlabeled_size=unlabeled_size,
            dataset=dataset,
            batch_size=private_batch_size
        )

        # Each client gets a copy of the global model
        model = get_model(dataset=dataset)
        model.load_state_dict(global_model.state_dict())

        clients.append({
            'id': client_id,
            'model': model,
            'trainloader': trainloader,
            'valloader': valloader,
            'public_dataset': public_dataset,
        })
        print()

    # Training loop
    consensus_labels = None

    for comm_round in range(1, num_communication_rounds + 1):
        print("\n" + "=" * 80)
        print(f"📊 COMMUNICATION ROUND {comm_round}/{num_communication_rounds}")
        print("=" * 80)

        # ========================================
        # STEP 1: LOCAL TRAINING
        # ========================================
        print(f"→ Each client will do {num_local_rounds} local epochs")

        # Update all clients with current global model
        for client in clients:
            client['model'].load_state_dict(global_model.state_dict())

        # Prepare pseudo-labeled loaders if we have consensus
        public_loaders = {}
        for client in clients:
            if consensus_labels is not None:
                pseudo_dataset = create_pseudo_labeled_dataset(
                    client['public_dataset'], consensus_labels
                )
                _, public_loader = combine_with_pseudo_labels(
                    client['trainloader'], pseudo_dataset, public_batch_size
                )
                public_loaders[client['id']] = public_loader
            else:
                public_loaders[client['id']] = None

        # Train all clients locally
        client_metrics = {}
        for client in clients:
            client_id = client['id']

            # Local training with mixed batches (private + pseudo-labeled)
            train_loss, train_acc = train(
                client['model'],
                client['trainloader'],
                epochs=num_local_rounds,
                device=device,
                optimizer_name=optimizer,
                learning_rate=learning_rate,
                public_loader=public_loaders[client_id]
            )

            # Evaluate
            test_loss, test_acc = test(
                client['model'], client['valloader'], device
            )

            client_metrics[client_id] = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc
            }

            print(f"  Client {client_id}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

        # ========================================
        # STEP 2: MODEL AGGREGATION (FedAvg)
        # ========================================
        print("\n🔄 AGGREGATING CLIENT MODELS (FedAvg)...")

        client_states = [get_state_dict(client['model']) for client in clients]
        aggregated_state = average_state_dicts(client_states)
        global_model.load_state_dict(aggregated_state)

        print("✓ Global model updated with averaged client weights")

        # ========================================
        # STEP 3: MAJORITY VOTING ON UNLABELED DATA
        # ========================================
        print("\n🗳️  MAJORITY VOTING ON PUBLIC DATASET...")

        # Each client predicts on public data using their local model
        client_predictions = {}
        for client in clients:
            predictions = predict_on_unlabeled(
                client['model'], client['public_dataset'], device
            )
            client_predictions[client['id']] = predictions

        # Perform majority voting
        consensus_labels = majority_vote(client_predictions)

        # ========================================
        # STEP 4: SUMMARY
        # ========================================
        print_summary(consensus_labels, client_predictions, client_metrics, comm_round)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)


def majority_vote(client_predictions):
    """Perform majority voting on client predictions"""
    num_samples = len(next(iter(client_predictions.values())))
    consensus_labels = []

    for sample_idx in range(num_samples):
        votes = [client_predictions[cid][sample_idx]
                 for cid in sorted(client_predictions.keys())]
        vote_counts = Counter(votes)
        winner = vote_counts.most_common(1)[0][0]
        consensus_labels.append(winner)

    return consensus_labels


def print_summary(consensus, client_predictions, client_metrics, comm_round):
    """Print summary of round"""
    num_samples = len(consensus)

    # Calculate agreement statistics
    agreements = []
    for sample_idx in range(num_samples):
        votes = [client_predictions[cid][sample_idx]
                 for cid in sorted(client_predictions.keys())]
        vote_counts = Counter(votes)
        winner_votes = max(vote_counts.values())
        agreement = winner_votes / len(votes)
        agreements.append(agreement)

    print("-" * 80)
    print(f"Round {comm_round} Summary:")
    print("-" * 80)

    print(f"\nMajority Voting Results:")
    print(f"  Mean Agreement: {np.mean(agreements):.3f}")
    print(f"  Min Agreement:  {np.min(agreements):.3f}")
    print(f"  Max Agreement:  {np.max(agreements):.3f}")

    test_accs = [m["test_acc"] for m in client_metrics.values()]
    test_losses = [m["test_loss"] for m in client_metrics.values()]

    print(f"\nClient Performance (after aggregation):")
    print(f"  Average Test Loss: {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")
    print(f"  Average Test Acc:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FedCT experiment with proper aggregation")
    parser.add_argument("--communication-rounds", type=int, default=15)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--local-rounds", type=int, default=3)
    parser.add_argument("--unlabeled", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "FashionMNIST"])
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--private-batch-size", type=int, default=64)
    parser.add_argument("--public-batch-size", type=int, default=32)

    args = parser.parse_args()

    run_fedcot_experiment(
        num_communication_rounds=args.communication_rounds,
        num_clients=args.clients,
        num_local_rounds=args.local_rounds,
        unlabeled_size=args.unlabeled,
        dataset=args.dataset,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        private_batch_size=args.private_batch_size,
        public_batch_size=args.public_batch_size
    )