#!/usr/bin/env python3
"""
Direct execution script for FedDCT that bypasses Ray simulation issues.
This runs the clients sequentially without Ray's distributed backend.
"""

import argparse
import sys
from pathlib import Path

# Add the project to the path
sys.path.insert(0, str(Path(__file__).parent))

from fedcot.task import get_model, load_data, train, test, predict_on_unlabeled, create_pseudo_labeled_dataset, combine_with_pseudo_labels
from fedcot.server_app import FedDCTStrategy
from collections import Counter
import torch
import numpy as np


def run_fedcot_experiment(num_communication_rounds=20, num_clients=5, num_local_rounds=5, unlabeled_size=500,
                         dataset="CIFAR10", optimizer="SGD", learning_rate=0.01,
                         private_batch_size=32, public_batch_size=32):
    """
    Run FedDCT experiment without Ray simulation
    
    Args:
        num_communication_rounds: Number of communication rounds
        num_clients: Number of clients
        num_local_rounds: Number of local rounds (passes through dataset) before each communication
        unlabeled_size: Size of public unlabeled dataset
        dataset: Dataset to use - "CIFAR10" or "FashionMNIST"
        optimizer: Optimizer to use - "SGD" or "Adam"
        learning_rate: Learning rate for optimizer
        private_batch_size: Batch size for private data sampling
        public_batch_size: Batch size for public pseudo-labeled data sampling
    """
    
    print("=" * 80)
    print("FEDERATED CO-TRAINING (FedDCT) - DIRECT EXECUTION")
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
        
        model = get_model(dataset=dataset)
        clients.append({
            'id': client_id,
            'model': model,
            'trainloader': trainloader,
            'valloader': valloader,
            'public_dataset': public_dataset,
            'pseudo_loader': None
        })
        print()
    
    # Training loop
    consensus_labels = None
    local_round_counter = 0
    
    for comm_round in range(1, num_communication_rounds + 1):
        print("\n" + "=" * 80)
        print(f"📊 COMMUNICATION CYCLE {comm_round}/{num_communication_rounds}")
        print("=" * 80)
        print(f"→ Clients will do {num_local_rounds} local training rounds")
        print(f"→ Then share predictions on public dataset")
        print("=" * 80)
        
        # Store metrics from last local round for reporting at communication round
        last_train_losses = {}
        last_train_accs = {}
        
        # Perform num_local_rounds before each communication
        for local_round in range(1, num_local_rounds + 1):
            local_round_counter += 1
            
            # Update pseudo-labeled dataset for all clients if we have consensus
            public_loaders = {}
            for client in clients:
                client_id = client['id']
                if consensus_labels is not None:
                    pseudo_dataset = create_pseudo_labeled_dataset(
                        client['public_dataset'], consensus_labels
                    )
                    _, public_loader = combine_with_pseudo_labels(
                        client['trainloader'], pseudo_dataset, public_batch_size
                    )
                    public_loaders[client_id] = public_loader
                else:
                    public_loaders[client_id] = None
            
            # Train all clients for 1 epoch (no evaluation during local rounds)
            for client in clients:
                client_id = client['id']
                
                # Train for 1 epoch with mixed batch sampling
                train_loss, train_acc = train(
                    client['model'], 
                    client['trainloader'], 
                    epochs=1,  # Just 1 epoch per local round
                    device=device,
                    optimizer_name=optimizer, 
                    learning_rate=learning_rate,
                    public_loader=public_loaders[client_id]
                )
                
                # Store metrics from this round (will use last round's metrics at communication)
                last_train_losses[client_id] = train_loss
                last_train_accs[client_id] = train_acc
            
            # Just acknowledge training completion (no metrics shared during local rounds)
            print(f"✓ Local Round {local_round}/{num_local_rounds}: Training completed (metrics shared only at communication rounds)")
        
        # After num_local_rounds, do communication
        print("\n" + "=" * 80)
        print(f"🗳️  COMMUNICATION ROUND {comm_round}/{num_communication_rounds} - MAJORITY VOTING")
        print("=" * 80)
        
        # Collect predictions and evaluate all clients at communication round
        client_predictions = {}
        client_metrics = {}
        for client in clients:
            client_id = client['id']
            
            # Make predictions on public dataset
            predictions = predict_on_unlabeled(
                client['model'], client['public_dataset'], device
            )
            client_predictions[client_id] = predictions
            
            # Evaluate on test set (only at communication rounds)
            test_loss, test_acc = test(
                client['model'], client['valloader'], device
            )
            
            # Use metrics from last local training round
            client_metrics[client_id] = {
                'train_loss': last_train_losses[client_id],
                'train_acc': last_train_accs[client_id],
                'test_loss': test_loss,
                'test_acc': test_acc
            }
        
        # Perform majority voting
        consensus_labels = majority_vote(client_predictions)
        
        # Print summary
        print_summary(consensus_labels, client_predictions, client_metrics)
    
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


def print_summary(consensus, client_predictions, client_metrics):
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
    print("MAJORITY VOTING RESULTS:")
    print("-" * 80)
    print(f"Consensus Labels (L̄_t) - {num_samples} samples:")
    if num_samples <= 20:
        print(f"  {consensus}")
    else:
        print(f"  {consensus[:20]}... (showing first 20)")
    
    print(f"\nAgreement Statistics:")
    print(f"  Mean Agreement: {np.mean(agreements):.3f}")
    print(f"  Min Agreement:  {np.min(agreements):.3f}")
    print(f"  Max Agreement:  {np.max(agreements):.3f}")
    
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
    
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FedDCT experiment directly (no Ray)")
    parser.add_argument("--communication-rounds", type=int, default=20, help="Number of communication rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--local-rounds", type=int, default=5, help="Number of local rounds (passes through dataset) before communication")
    parser.add_argument("--unlabeled", type=int, default=500, help="Size of unlabeled dataset")
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "FashionMNIST"], 
                        help="Dataset to use (default: CIFAR10)")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"], 
                        help="Optimizer to use (default: SGD)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--private-batch-size", type=int, default=32, help="Batch size for private data (default: 32)")
    parser.add_argument("--public-batch-size", type=int, default=32, help="Batch size for public pseudo-labeled data (default: 32)")
    
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

