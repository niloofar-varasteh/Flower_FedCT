import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def parse_fedavg(log_path: Path):
    """Parse FedAvg log lines like:
       Round 7 - Loss: 0.9025 - Accuracy: 0.7250
    """
    rounds, losses, accs = [], [], []
    pat = re.compile(r"Round\s+(\d+)\s+-\s+Loss:\s+([0-9.]+)\s+-\s+Accuracy:\s+([0-9.]+)")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            r = int(m.group(1))
            loss = float(m.group(2))
            acc = float(m.group(3))
            rounds.append(r); losses.append(loss); accs.append(acc)
    return rounds, losses, accs

def parse_fedct(log_path: Path):
    """Parse FedCT lines like:
       ✓ Local Round 2/3 [Flower Round 41]: Train Loss: 1.5718, Train Acc: 0.4677, Test Loss: 1.6860, Test Acc: 0.4113
       همچنین تشخیص می‌دهد پایان هر cycle کجاست (وقتی a==b).
    """
    srounds, test_losses, test_accs = [], [], []
    cycle_rounds, cycle_accs = [], []

    pat = re.compile(
        r"Local Round\s+(\d+)/(\d+)\s+\[Flower Round\s+(\d+)\]:.*?Test Loss:\s+([0-9.]+),\s+Test Acc:\s+([0-9.]+)"
    )
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if not m:
            # برخی لاگ‌ها «✓» اول خط دارند؛ الگو بالا بدون آن هم کار می‌کند.
            continue
        a = int(m.group(1))         # local within cycle
        b = int(m.group(2))         # local per cycle
        R = int(m.group(3))         # Flower server round
        tl = float(m.group(4))
        ta = float(m.group(5))
        srounds.append(R)
        test_losses.append(tl)
        test_accs.append(ta)
        if a == b:
            cycle_rounds.append(R)
            cycle_accs.append(ta)

    return srounds, test_losses, test_accs, cycle_rounds, cycle_accs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fedavg-log", required=True, help="Path to FedAvg experiment.log")
    ap.add_argument("--fedct-log", required=True, help="Path to FedCT experiment.log")
    ap.add_argument("--out", default="fedct_vs_fedavg.png", help="Output image file")
    ap.add_argument("--title", default="FedCT vs FedAvg (CIFAR10, 5 clients)")
    args = ap.parse_args()

    fav_r, fav_l, fav_a = parse_fedavg(Path(args.fedavg_log))
    fct_r, fct_l, fct_a, fct_cR, fct_cA = parse_fedct(Path(args.fedct_log))

    plt.figure(figsize=(12,5))

    # ---- Loss ----
    ax1 = plt.subplot(1,2,1)
    if fct_r: ax1.plot(fct_r, fct_l, label="FedCT (per Flower round)")
    if fav_r: ax1.plot(fav_r, fav_l, label="FedAvg (per round)")
    ax1.set_title("Loss vs Round")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ---- Accuracy ----
    ax2 = plt.subplot(1,2,2)
    if fct_r: ax2.plot(fct_r, fct_a, label="FedCT (per Flower round)")
    if fav_r: ax2.plot(fav_r, fav_a, label="FedAvg (per round)")
    # مارکر روی پایان هر communication cycle در FedCT
    if fct_cR:
        ax2.scatter(fct_cR, fct_cA, s=28, marker="o", edgecolors="k", label="FedCT (end of cycle)")
    ax2.set_title("Accuracy vs Round")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(args.title, y=1.02)
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print(f"Saved plot -> {args.out}")

if __name__ == "__main__":
    main()
