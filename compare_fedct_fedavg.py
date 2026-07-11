#!/usr/bin/env python3
"""Compare FedCT and FedAvg experiment logs and save comparison plots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def latest_log(root: str, dataset: str, arch: str) -> Path | None:
    root_path = Path(root)
    if not root_path.exists():
        return None
    prefix = f"{dataset}_{arch}_"
    logs = [
        p
        for p in root_path.rglob("experiment.log")
        if p.parent.name.startswith(prefix)
    ]
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def parse_fedct(path: Path, metric_type: str):
    pattern = re.compile(
        r"Local Round\s+(\d+)/(\d+)\s+\[Flower Round\s+(\d+)\]:.*?"
        r"Train Loss:\s+([0-9.]+),\s+Train Acc:\s+([0-9.]+),\s+"
        r"Test Loss:\s+([0-9.]+),\s+Test Acc:\s+([0-9.]+)"
    )
    xs, losses, accs = [], [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        flower_round = int(match.group(3))
        train_loss = float(match.group(4))
        train_acc = float(match.group(5))
        test_loss = float(match.group(6))
        test_acc = float(match.group(7))
        xs.append(flower_round - 1)
        if metric_type == "train":
            losses.append(train_loss)
            accs.append(train_acc)
        else:
            losses.append(test_loss)
            accs.append(test_acc)
    return xs, losses, accs


def parse_fedavg(path: Path, metric_type: str):
    pattern = re.compile(
        r"\[LOCAL\]\s+epoch=(\d+)/(\d+)\s+"
        r"train_acc=([0-9.]+)\s+train_loss=([0-9.]+)\s+"
        r"test_acc=([0-9.]+)\s+test_loss=([0-9.]+)"
    )
    xs, losses, accs = [], [], []
    comm_round = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        round_match = re.search(r"Aggregation Round\s+(\d+)/", line)
        if round_match:
            comm_round = int(round_match.group(1))
            continue

        match = pattern.search(line)
        if not match:
            continue
        epoch = int(match.group(1))
        local_total = int(match.group(2))
        train_acc = float(match.group(3))
        train_loss = float(match.group(4))
        test_acc = float(match.group(5))
        test_loss = float(match.group(6))
        step = (comm_round - 1) * local_total + (epoch - 1)
        xs.append(step)
        if metric_type == "train":
            losses.append(train_loss)
            accs.append(train_acc)
        else:
            losses.append(test_loss)
            accs.append(test_acc)
    return xs, losses, accs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["CIFAR10", "FashionMNIST"])
    parser.add_argument("--arch", required=True, choices=["LightCNN", "ResNet18"])
    parser.add_argument("--metric-type", default="test", choices=["train", "test"])
    parser.add_argument("--fedcot-log")
    parser.add_argument("--fedavg-log")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="FedCT vs FedAvg")
    args = parser.parse_args()

    fedct_log = Path(args.fedcot_log) if args.fedcot_log else latest_log("logs", args.dataset, args.arch)
    fedavg_log = Path(args.fedavg_log) if args.fedavg_log else latest_log("logs_fedavg", args.dataset, args.arch)

    if not fedct_log or not fedct_log.exists():
        raise SystemExit(f"FedCT log not found for {args.dataset} {args.arch}")
    if not fedavg_log or not fedavg_log.exists():
        raise SystemExit(f"FedAvg log not found for {args.dataset} {args.arch}")

    print(f"Using FedCT  log: {fedct_log}")
    print(f"Using FedAvg log: {fedavg_log}")

    fct_x, fct_loss, fct_acc = parse_fedct(fedct_log, args.metric_type)
    fav_x, fav_loss, fav_acc = parse_fedavg(fedavg_log, args.metric_type)

    if not fct_x:
        raise SystemExit("No FedCT points parsed. Check whether the FedCT run completed enough local rounds.")
    if not fav_x:
        raise SystemExit("No FedAvg points parsed. Check whether [LOCAL] lines exist in the FedAvg log.")

    metric_label = "Train" if args.metric_type == "train" else "Test"
    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(fct_x, fct_loss, "o-", label="FedCT", linewidth=2, markersize=4)
    ax1.plot(fav_x, fav_loss, "s-", label="FedAvg", linewidth=2, markersize=4)
    ax1.set_xlabel("Local Training Step")
    ax1.set_ylabel(f"{metric_label} Loss")
    ax1.set_title(f"{metric_label} Loss Comparison", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(fct_x, fct_acc, "o-", label="FedCT", linewidth=2, markersize=4, color="#2ecc71")
    ax2.plot(fav_x, fav_acc, "s-", label="FedAvg", linewidth=2, markersize=4, color="#e74c3c")
    ax2.set_xlabel("Local Training Step")
    ax2.set_ylabel(f"{metric_label} Accuracy")
    ax2.set_title(f"{metric_label} Accuracy Comparison", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f"{args.title} - Metric: {args.metric_type}", y=1.02, fontweight="bold")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"Saved plot -> {args.out}")


if __name__ == "__main__":
    main()
