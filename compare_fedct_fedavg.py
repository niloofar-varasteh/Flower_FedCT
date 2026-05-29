#!/usr/bin/env python3
"""
Compare FedCT and FedAvg results from experiment logs.
Focuses on local round comparison for fair evaluation.
"""

import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from statistics import mean


def latest_log(root: str):
    """Return most recent experiment.log under root/**/experiment.log, or None."""
    root_path = Path(root)
    if not root_path.exists():
        return None
    cands = sorted(root_path.rglob("experiment.log"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def parse_fedavg_local(path: Path, metric_type="test"):
    """
    Parse per-local-epoch metrics from FedAvg logs.
    
    Supports:
    [LOCAL] epoch=E/T acc=0.1234 loss=0.5678   ← old format (test metrics)
    [LOCAL] epoch=E/T train_acc=... train_loss=... test_acc=... test_loss=...  ← new format (both train and test)
    
    Args:
        metric_type: "train" or "test" to select which metrics to return
    """
    r_local, accs, losses = [], [], []
    
    # Try new format first (with train/test distinction)
    pat_new = re.compile(r"\[LOCAL\]\s*epoch=(\d+)/(\d+)\s+train_acc=([0-9.]+)\s+train_loss=([0-9.]+)\s+test_acc=([0-9.]+)\s+test_loss=([0-9.]+)")
    # Fallback to old format (test metrics only)
    pat_old = re.compile(r"\[LOCAL\]\s*epoch=(\d+)/(\d+)\s+acc=([0-9.]+)\s+loss=([0-9.]+)")
    
    # Also try to extract communication round info from context
    comm_round = 0
    local_rounds_per_comm = None
    
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        
        # Check for communication round info
        comm_match = re.search(r"Aggregation Round\s+(\d+)/", s)
        if comm_match:
            comm_round = int(comm_match.group(1))
            continue
        
        # Try new format first
        m_new = pat_new.search(s)
        if m_new:
            e, T = int(m_new.group(1)), int(m_new.group(2))
            train_acc = float(m_new.group(3))
            train_loss = float(m_new.group(4))
            test_acc = float(m_new.group(5))
            test_loss = float(m_new.group(6))
            
            if local_rounds_per_comm is None:
                local_rounds_per_comm = T
            
            step = (comm_round - 1) * local_rounds_per_comm + (e - 1)
            r_local.append(step)
            
            if metric_type == "train":
                accs.append(train_acc)
                losses.append(train_loss)
            else:  # test
                accs.append(test_acc)
                losses.append(test_loss)
            continue
        
        # Try old format (backward compatibility)
        m_old = pat_old.search(s)
        if m_old:
            e, T = int(m_old.group(1)), int(m_old.group(2))
            acc = float(m_old.group(3))
            los = float(m_old.group(4))
            
            if local_rounds_per_comm is None:
                local_rounds_per_comm = T
            
            step = (comm_round - 1) * local_rounds_per_comm + (e - 1)
            r_local.append(step)
            # Old format only has test metrics
            if metric_type == "test":
                accs.append(acc)
                losses.append(los)
            # If train requested but only test available, skip or use test (user will see warning)
            elif metric_type == "train":
                # Skip this point - train metrics not available in old format
                continue
    
    return r_local, losses, accs


def parse_fedct_local(path: Path, metric_type="test"):
    """
    Parse every local step for FedCT (handles both direct and Flower logs).
    
    Pattern handles:
    Local Round 1/3 [Flower Round 1]: Train Loss: X, Train Acc: Y, Test Loss: X, Test Acc: Y
    
    Args:
        metric_type: "train" or "test" to select which metrics to return
    """
    # Pattern matches: Train Loss: X, Train Acc: Y, Test Loss: X, Test Acc: Y
    pat = re.compile(
        r"Local Round\s+(\d+)/(\d+)\s+\[Flower Round\s+(\d+)\]:.*?Train Loss:\s+([0-9.]+),\s+Train Acc:\s+([0-9.]+),\s+Test Loss:\s+([0-9.]+),\s+Test Acc:\s+([0-9.]+)"
    )
    xs, losses, accs = [], [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            local_round, total_local, comm_round = int(m.group(1)), int(m.group(2)), int(m.group(3))
            train_loss = float(m.group(4))
            train_acc = float(m.group(5))
            test_loss = float(m.group(6))
            test_acc = float(m.group(7))
            
            # Global step: (comm_round - 1) * total_local + local_round - 1
            step = (comm_round - 1) * total_local + (local_round - 1)
            xs.append(step)
            
            if metric_type == "train":
                losses.append(train_loss)
                accs.append(train_acc)
            else:  # test
                losses.append(test_loss)
                accs.append(test_acc)
    return xs, losses, accs


def parse_fedct_comm(path: Path):
    """Parse FedCT communication-round logs (only at end of each communication round)."""
    srounds, test_losses, test_accs = [], [], []
    cycle_rounds, cycle_accs = [], []
    pat = re.compile(
        r"Local Round\s+(\d+)/(\d+)\s+\[Flower Round\s+(\d+)\]:.*?Test Loss:\s+([0-9.]+),\s+Test Acc:\s+([0-9.]+)"
    )
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            local_round, total_local, R = int(m.group(1)), int(m.group(2)), int(m.group(3))
            tl, ta = float(m.group(4)), float(m.group(5))
            srounds.append(R)
            test_losses.append(tl)
            test_accs.append(ta)
            # Only keep the last local round of each communication round
            if local_round == total_local:
                cycle_rounds.append(R)
                cycle_accs.append(ta)
    
    loss_by_round = {r: l for r, l in zip(srounds, test_losses)}
    r_use = list(range(1, len(cycle_rounds) + 1))
    a_use = cycle_accs
    l_use = [loss_by_round.get(cr, test_losses[-1]) for cr in cycle_rounds]
    return r_use, l_use, a_use


def main():
    ap = argparse.ArgumentParser(description="Compare FedCT vs FedAvg results")
    ap.add_argument("--fedavg-log", type=str, help="Path to FedAvg experiment.log (optional)")
    ap.add_argument("--fedcot-log", type=str, help="Path to FedCT experiment.log (optional)")
    ap.add_argument("--out", type=str, default="FashionMNIST_LightCNN_test.png")
    ap.add_argument("--title", type=str, default="FedCT vs FedAvg")
    ap.add_argument("--mode", choices=["comm", "local"], default="local",
                    help="comm = per communication round, local = per local epoch/step")
    ap.add_argument("--metric-type", choices=["train", "test"], default="test",
                    help="train = plot train metrics, test = plot test metrics (default: test)")
    args = ap.parse_args()
    
    fav_path = Path(args.fedavg_log) if args.fedavg_log else latest_log("logs_fedavg")
    fct_path = Path(args.fedcot_log) if args.fedcot_log else latest_log("logs")
    
    if not fav_path or not fav_path.exists():
        raise SystemExit("FedAvg log not found. Pass --fedavg-log or put logs under logs_fedavg/**/experiment.log")
    if not fct_path or not fct_path.exists():
        raise SystemExit("FedCT log not found. Pass --fedcot-log or put logs under logs/**/experiment.log")
    
    print(f"\nUsing FedAvg log: {fav_path}")
    print(f"Using FedCT  log: {fct_path}\n")
    
    if args.mode == "comm":
        # For communication rounds, we need to extract only the last local round of each comm round
        # Note: comm mode currently only supports test metrics
        fav_x, fav_l, fav_a = parse_fedavg_local(fav_path, metric_type="test")
        fct_x, fct_l, fct_a = parse_fedct_comm(fct_path)
        x_label = "Communication Round"
        metric_label = "Test"
    else:  # local
        fav_x, fav_l, fav_a = parse_fedavg_local(fav_path, metric_type=args.metric_type)
        fct_x, fct_l, fct_a = parse_fedct_local(fct_path, metric_type=args.metric_type)
        x_label = "Local Training Step"
        metric_label = "Train" if args.metric_type == "train" else "Test"
    
    if not fav_x:
        raise SystemExit("No FedAvg points parsed. Check log format (are [LOCAL] lines enabled?).")
    if not fct_x:
        raise SystemExit("No FedCT points parsed. Check FedCT log format.")
    
    # Warn if train metrics requested but not available
    if args.metric_type == "train" and not fav_l:
        print("⚠ WARNING: Train metrics not found in FedAvg log. This may be an old log format.")
        print("   Re-run FedAvg with the updated code to get train metrics.")
        print("   Falling back to test metrics...")
        fav_x, fav_l, fav_a = parse_fedavg_local(fav_path, metric_type="test")
        metric_label = "Test"
    
    # Create comparison plot
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(fct_x, fct_l, "o-", label="FedCT", linewidth=2, markersize=4)
    ax1.plot(fav_x, fav_l, "s-", label="FedAvg", linewidth=2, markersize=4)
    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel(f"{metric_label} Loss", fontsize=11)
    ax1.set_title(f"{metric_label} Loss Comparison", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Accuracy plot
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(fct_x, fct_a, "o-", label="FedCT", linewidth=2, markersize=4, color='#2ecc71')
    ax2.plot(fav_x, fav_a, "s-", label="FedAvg", linewidth=2, markersize=4, color='#e74c3c')
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel(f"{metric_label} Accuracy", fontsize=11)
    ax2.set_title(f"{metric_label} Accuracy Comparison", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add final accuracy values as text
    if fct_a and fav_a:
        final_fedct = fct_a[-1]
        final_fedavg = fav_a[-1]
        improvement = ((final_fedct - final_fedavg) / final_fedavg) * 100
        ax2.text(0.02, 0.98, f"Final:\nFedCT: {final_fedct:.3f}\nFedAvg: {final_fedavg:.3f}\nImprovement: {improvement:+.1f}%",
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"{args.title} — Mode: {args.mode}, Metric: {metric_label.lower()}", y=1.02, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.out, dpi=220, bbox_inches='tight')
    print(f"✓ Saved plot -> {args.out}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print(f"Comparison Summary ({metric_label} Metrics)")
    print("=" * 80)
    if fct_a and fav_a:
        print(f"FedCT Final {metric_label} Accuracy:  {fct_a[-1]:.4f}")
        print(f"FedAvg Final {metric_label} Accuracy: {fav_a[-1]:.4f}")
        improvement = ((fct_a[-1] - fav_a[-1]) / fav_a[-1]) * 100
        print(f"Improvement:           {improvement:+.2f}%")
        print(f"FedCT Max {metric_label} Accuracy:    {max(fct_a):.4f}")
        print(f"FedAvg Max {metric_label} Accuracy:   {max(fav_a):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

